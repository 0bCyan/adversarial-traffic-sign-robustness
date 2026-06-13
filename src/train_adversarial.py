import argparse
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from src.attacks.methods import fgsm_attack, pgd_attack
from src.data.gtsrb_labels import GTSRB_LABELS
from src.models.classifiers import build_model

NORMALIZE_MEAN = torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1)
NORMALIZE_STD = torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def ensure_output_dirs(result_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": result_dir,
        "logs": result_dir / "logs",
        "metrics": result_dir / "metrics",
        "figures": result_dir / "figures",
        "samples": result_dir / "samples",
        "checkpoints": result_dir / "checkpoints",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def make_train_loaders(config: dict, seed: int) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    root = data_cfg.get("root") or data_cfg.get("raw_dir", "data/raw")
    if str(root).replace("\\", "/").endswith("/gtsrb"):
        root = str(Path(root).parent)
    image_size = int(data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    validation_ratio = float(data_cfg.get("validation_ratio", 0.15))

    train_aug = GTSRB(root=root, split="train", transform=build_train_transform(image_size), download=bool(data_cfg.get("download", True)))
    train_eval = GTSRB(root=root, split="train", transform=build_eval_transform(image_size), download=False)
    labels = [int(label) for _, label in train_aug._samples]
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        stratify=labels,
    )
    train_subset = Subset(train_aug, train_idx.tolist())
    val_subset = Subset(train_eval, val_idx.tolist())
    return (
        make_loader(train_subset, batch_size, num_workers, shuffle=True),
        make_loader(val_subset, batch_size, num_workers, shuffle=False),
    )


def make_eval_loader(config: dict, dirs: dict[str, Path]) -> DataLoader:
    data_cfg = config["data"]
    dataset = GTSRB(
        root=data_cfg.get("root", "data/raw"),
        split="test",
        transform=build_eval_transform(int(data_cfg["image_size"])),
        download=False,
    )
    selected_path = data_cfg.get("selected_indices")
    if selected_path and Path(selected_path).exists():
        selected_indices = pd.read_csv(selected_path)["index"].astype(int).tolist()
        dataset = Subset(dataset, selected_indices)
    else:
        max_eval_samples = int(data_cfg.get("max_eval_samples", 0))
        if max_eval_samples and max_eval_samples < len(dataset):
            labels = [int(label) for _, label in dataset._samples]
            indices = np.arange(len(dataset))
            _, selected_indices = train_test_split(
                indices,
                test_size=max_eval_samples,
                random_state=int(config.get("seed", 42)),
                stratify=labels,
            )
            selected_indices = sorted(selected_indices.tolist())
            pd.DataFrame({"index": selected_indices}).to_csv(
                dirs["metrics"] / "selected_eval_indices.csv",
                index=False,
                encoding="utf-8-sig",
            )
            dataset = Subset(dataset, selected_indices)
    return make_loader(
        dataset,
        int(data_cfg["batch_size"]),
        int(data_cfg.get("num_workers", 0)),
        shuffle=False,
    )


def clamp_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = NORMALIZE_MEAN.to(device)
    std = NORMALIZE_STD.to(device)
    return (0.0 - mean) / std, (1.0 - mean) / std


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


def load_initial_model(config: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(
        config["model"]["name"],
        num_classes=int(config["data"]["num_classes"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    )
    init_checkpoint = config["model"].get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def generate_training_adversarial(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    data_min, data_max = clamp_bounds(device)
    adversarial = fgsm_attack(model, images, labels, epsilon, data_min, data_max)
    model.train(was_training)
    return adversarial


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict,
) -> tuple[float, float]:
    model.train()
    losses = []
    correct = 0
    total = 0
    defense_cfg = config["defense"]
    epsilon = float(defense_cfg["epsilon"])
    clean_weight = float(defense_cfg.get("clean_weight", 0.5))
    adversarial_weight = float(defense_cfg.get("adversarial_weight", 0.5))

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adversarial = generate_training_adversarial(model, images, labels, epsilon, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            clean_logits = model(images)
            adv_logits = model(adversarial)
            clean_loss = F.cross_entropy(clean_logits, labels)
            adv_loss = F.cross_entropy(adv_logits, labels)
            loss = clean_weight * clean_loss + adversarial_weight * adv_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        preds = adv_logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return float(np.mean(losses)), correct / total


@torch.no_grad()
def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(F.cross_entropy(logits, labels).detach().cpu()))
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return float(np.mean(losses)), accuracy_score(y_true, y_pred)


def plot_training_curves(log_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train adversarial loss")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="Validation clean loss")
    ax.set_title("Adversarial Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "adversarial_training_loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_adv_acc"], label="Train adversarial accuracy")
    ax.plot(log_df["epoch"], log_df["val_clean_acc"], label="Validation clean accuracy")
    ax.set_title("Adversarial Training Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "adversarial_training_accuracy_curve.png", dpi=200)
    plt.close(fig)


def make_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_name: str,
    epsilon: float,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    data_min, data_max = clamp_bounds(device)
    if attack_name == "clean":
        return images
    if attack_name == "fgsm":
        return fgsm_attack(model, images, labels, epsilon, data_min, data_max)
    if attack_name == "pgd":
        eval_cfg = config["evaluation"]
        return pgd_attack(
            model,
            images,
            labels,
            epsilon,
            float(eval_cfg.get("pgd_alpha", 0.005)),
            int(eval_cfg.get("pgd_steps", 7)),
            data_min,
            data_max,
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


@torch.no_grad()
def predict(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def tensor_to_pil(tensor: torch.Tensor, size: int = 112) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def save_robust_grid(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        return
    cols = min(3, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    image_size = 112
    cell_w = 270
    cell_h = image_size + 76
    title_h = 44
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 14), title, fill="black", font=font)

    for idx, case in enumerate(cases):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = title_h + row * cell_h
        canvas.paste(tensor_to_pil(case["clean"], image_size), (x + 10, y))
        canvas.paste(tensor_to_pil(case["adversarial"], image_size), (x + 146, y))
        draw.text((x + 44, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 174, y + image_size + 5), "Attack", fill="red", font=font)
        label = case["label"]
        draw.text((x + 10, y + image_size + 25), f"True: {label} {GTSRB_LABELS[label][:22]}", fill="black", font=font)
        draw.text((x + 10, y + image_size + 43), f"Pred: {case['pred']} conf={case['conf']:.3f}", fill="green", font=font)
    canvas.save(output_path)


def evaluate_robustness(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    model.eval()
    rows = []
    sample_cases = []
    max_cases = int(config["outputs"].get("max_cases_per_grid", 9))

    for attack_name in config["evaluation"]["attacks"]:
        epsilons = [0.0] if attack_name == "clean" else config["evaluation"]["epsilons"]
        for epsilon in epsilons:
            epsilon = float(epsilon)
            total = 0
            correct = 0
            confidences = []
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                adversarial = make_attack(model, images, labels, attack_name, epsilon, config, device)
                pred, conf = predict(model, adversarial)
                total += int(labels.numel())
                correct_mask = pred == labels
                correct += int(correct_mask.sum().item())
                confidences.extend(conf.detach().cpu().numpy().tolist())

                if attack_name == "pgd" and np.isclose(epsilon, 0.03) and len(sample_cases) < max_cases:
                    clean_rgb = denormalize(images)
                    adv_rgb = denormalize(adversarial)
                    for idx in torch.where(correct_mask)[0].detach().cpu().numpy().tolist():
                        if len(sample_cases) >= max_cases:
                            break
                        sample_cases.append(
                            {
                                "clean": clean_rgb[idx].detach().cpu(),
                                "adversarial": adv_rgb[idx].detach().cpu(),
                                "label": int(labels[idx].detach().cpu()),
                                "pred": int(pred[idx].detach().cpu()),
                                "conf": float(conf[idx].detach().cpu()),
                            }
                        )

            row = {
                "model": "adversarial_training",
                "attack": attack_name,
                "epsilon": epsilon,
                "accuracy": correct / total,
                "mean_confidence": float(np.mean(confidences)),
                "total_images": total,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    save_robust_grid(
        sample_cases,
        dirs["samples"] / "adversarial_training_pgd_eps003_robust_examples.png",
        "Adversarially Trained Model, PGD epsilon=0.03",
    )
    return pd.DataFrame(rows)


def plot_robustness(robust_df: pd.DataFrame, baseline_metrics_path: Path, figure_dir: Path) -> None:
    baseline_df = pd.read_csv(baseline_metrics_path) if baseline_metrics_path.exists() else pd.DataFrame()

    attack_rows = robust_df[robust_df["attack"] != "clean"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, group in attack_rows.groupby("attack"):
        ax.plot(group["epsilon"], group["accuracy"], marker="o", label=f"adv training {attack}")
        if not baseline_df.empty:
            baseline_group = baseline_df[baseline_df["attack"] == attack]
            ax.plot(
                baseline_group["epsilon"],
                baseline_group["adversarial_accuracy"],
                marker="x",
                linestyle="--",
                label=f"baseline {attack}",
            )
    ax.set_title("Robust Accuracy: Baseline vs Adversarial Training")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "adversarial_training_robust_accuracy_curve.png", dpi=200)
    plt.close(fig)

    eps = 0.03
    labels = []
    values = []
    for attack in ["fgsm", "pgd"]:
        if not baseline_df.empty:
            baseline_value = baseline_df[(baseline_df["attack"] == attack) & (np.isclose(baseline_df["epsilon"], eps))]
            if not baseline_value.empty:
                labels.append(f"baseline {attack}")
                values.append(float(baseline_value.iloc[0]["adversarial_accuracy"]))
        adv_value = robust_df[(robust_df["attack"] == attack) & (np.isclose(robust_df["epsilon"], eps))]
        if not adv_value.empty:
            labels.append(f"adv train {attack}")
            values.append(float(adv_value.iloc[0]["accuracy"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("Robust Accuracy Comparison at Epsilon=0.03")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "adversarial_training_eps003_bar.png", dpi=200)
    plt.close(fig)


def build_strategy_comparison(
    robust_df: pd.DataFrame,
    baseline_metrics_path: Path,
    input_defense_metrics_path: Path,
    jpeg_quality: int,
    metrics_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    baseline_df = pd.read_csv(baseline_metrics_path) if baseline_metrics_path.exists() else pd.DataFrame()
    input_df = pd.read_csv(input_defense_metrics_path) if input_defense_metrics_path.exists() else pd.DataFrame()
    rows = []

    if not baseline_df.empty:
        clean_accuracy = float(baseline_df["clean_accuracy"].iloc[0])
        rows.append(
            {
                "strategy": "baseline",
                "attack": "clean",
                "epsilon": 0.0,
                "accuracy": clean_accuracy,
                "note": "clean accuracy from baseline attack evaluation",
            }
        )
        for row in baseline_df.itertuples(index=False):
            rows.append(
                {
                    "strategy": "baseline",
                    "attack": row.attack,
                    "epsilon": float(row.epsilon),
                    "accuracy": float(row.adversarial_accuracy),
                    "note": "no defense",
                }
            )

    if not input_df.empty:
        jpeg_rows = input_df[input_df["defense"] == "jpeg_compression"]
        for row in jpeg_rows.itertuples(index=False):
            rows.append(
                {
                    "strategy": f"jpeg_compression_q{jpeg_quality}",
                    "attack": row.attack,
                    "epsilon": float(row.epsilon),
                    "accuracy": float(row.defended_accuracy),
                    "note": "input preprocessing defense",
                }
            )

    for row in robust_df.itertuples(index=False):
        rows.append(
            {
                "strategy": "adversarial_training",
                "attack": row.attack,
                "epsilon": float(row.epsilon),
                "accuracy": float(row.accuracy),
                "note": "FGSM adversarially trained model",
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(metrics_dir / "defense_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    if comparison_df.empty:
        return comparison_df

    eps = 0.03
    eps_df = comparison_df[(comparison_df["attack"].isin(["fgsm", "pgd"])) & np.isclose(comparison_df["epsilon"], eps)]
    if not eps_df.empty:
        labels = [f"{row.attack}\n{row.strategy}" for row in eps_df.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, eps_df["accuracy"].tolist())
        ax.set_title("Defense Strategy Comparison at Epsilon=0.03")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "defense_strategy_comparison_eps003.png", dpi=200)
        plt.close(fig)

    clean_df = comparison_df[comparison_df["attack"] == "clean"]
    robust_eps = comparison_df[(comparison_df["attack"].isin(["fgsm", "pgd"])) & np.isclose(comparison_df["epsilon"], eps)]
    if not clean_df.empty and not robust_eps.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in sorted(comparison_df["strategy"].unique()):
            clean_value = clean_df[clean_df["strategy"] == strategy]
            robust_value = robust_eps[robust_eps["strategy"] == strategy]
            if clean_value.empty or robust_value.empty:
                continue
            ax.scatter(
                float(clean_value.iloc[0]["accuracy"]),
                float(robust_value["accuracy"].mean()),
                label=strategy,
                s=80,
            )
        ax.set_title("Clean Accuracy vs Mean Robust Accuracy")
        ax.set_xlabel("Clean accuracy")
        ax.set_ylabel("Mean robust accuracy at epsilon=0.03")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "clean_vs_robust_tradeoff.png", dpi=200)
        plt.close(fig)

    return comparison_df


def copy_report_assets(dirs: dict[str, Path], report_figure_dir: Path, report_table_dir: Path) -> None:
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "adversarial_training_loss_curve.png": "fig_20_adv_training_loss_curve.png",
        "adversarial_training_accuracy_curve.png": "fig_21_adv_training_accuracy_curve.png",
        "adversarial_training_robust_accuracy_curve.png": "fig_22_adv_training_robust_curve.png",
        "adversarial_training_eps003_bar.png": "fig_23_adv_training_eps003_bar.png",
        "defense_strategy_comparison_eps003.png": "fig_24_defense_strategy_comparison.png",
        "clean_vs_robust_tradeoff.png": "fig_25_clean_vs_robust_tradeoff.png",
    }
    for src_name, dst_name in figure_map.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)

    sample = dirs["samples"] / "adversarial_training_pgd_eps003_robust_examples.png"
    if sample.exists():
        shutil.copy2(sample, report_figure_dir / "fig_26_adv_training_robust_examples.png")

    for src_name, dst_name in {
        "train_log.csv": "table_08_adv_training_train_log.csv",
        "robust_metrics.csv": "table_09_adv_training_robust_metrics.csv",
        "defense_strategy_comparison.csv": "table_10_defense_strategy_comparison.csv",
    }.items():
        src = dirs["metrics"] / src_name
        if src.exists():
            shutil.copy2(src, report_table_dir / dst_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a GTSRB classifier with FGSM adversarial training.")
    parser.add_argument("--config", type=Path, default=Path("configs/defense_adversarial_training.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    result_dir = args.output_dir or Path(config["outputs"]["result_dir"])
    dirs = ensure_output_dirs(result_dir)
    shutil.copy2(args.config, dirs["root"] / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    write_json(
        dirs["root"] / "run_info.json",
        {
            "experiment_name": config.get("experiment_name"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
        },
    )

    train_loader, val_loader = make_train_loaders(config, seed)
    eval_loader = make_eval_loader(config, dirs)
    model = load_initial_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = -1.0
    history = []
    start = time.time()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, train_adv_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        val_loss, val_clean_acc = evaluate_clean(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_adv_acc": train_adv_acc,
            "val_loss": val_loss,
            "val_clean_acc": val_clean_acc,
            "lr": lr,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{config['training']['epochs']} "
            f"train_loss={train_loss:.4f} train_adv_acc={train_adv_acc:.4f} "
            f"val_loss={val_loss:.4f} val_clean_acc={val_clean_acc:.4f}"
        )
        if val_clean_acc > best_val_acc:
            best_val_acc = val_clean_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                dirs["checkpoints"] / "best_model.pth",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_acc": best_val_acc,
            "epoch": int(config["training"]["epochs"]),
        },
        dirs["checkpoints"] / "last_model.pth",
    )

    train_log = pd.DataFrame(history)
    train_log.to_csv(dirs["metrics"] / "train_log.csv", index=False, encoding="utf-8-sig")
    plot_training_curves(train_log, dirs["figures"])

    checkpoint = torch.load(dirs["checkpoints"] / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    robust_df = evaluate_robustness(model, eval_loader, config, device, dirs)
    robust_df["training_seconds"] = float(time.time() - start)
    robust_df["best_epoch"] = int(checkpoint["epoch"])
    robust_df["best_val_accuracy"] = float(checkpoint["best_val_acc"])
    robust_df.to_csv(dirs["metrics"] / "robust_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["metrics"] / "robust_summary.json", {"metrics": robust_df.to_dict(orient="records")})

    plot_robustness(
        robust_df,
        Path(config["evaluation"].get("baseline_attack_metrics", "results/02_attack/fgsm_pgd/metrics/attack_metrics.csv")),
        dirs["figures"],
    )
    build_strategy_comparison(
        robust_df,
        Path(config["evaluation"].get("baseline_attack_metrics", "results/02_attack/fgsm_pgd/metrics/attack_metrics.csv")),
        Path(config["evaluation"].get("input_defense_metrics", "results/03_defense/input_preprocessing/metrics/input_defense_metrics.csv")),
        int(config["evaluation"].get("jpeg_quality", 75)),
        dirs["metrics"],
        dirs["figures"],
    )
    copy_report_assets(
        dirs,
        Path(config["outputs"].get("report_figure_dir", "reports/figures")),
        Path(config["outputs"].get("report_table_dir", "reports/tables")),
    )
    with (dirs["logs"] / "adversarial_training.log").open("w", encoding="utf-8") as f:
        f.write(robust_df.to_string(index=False))
        f.write("\n")
    print(f"Adversarial training completed: {result_dir}")


if __name__ == "__main__":
    main()

