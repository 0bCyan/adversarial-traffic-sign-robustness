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
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from tqdm import tqdm

from src.data.gtsrb_labels import GTSRB_LABELS
from src.models.classifiers import build_model, count_parameters


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


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
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
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def make_loaders(config: dict, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, GTSRB]:
    data_cfg = config["data"]
    root = data_cfg.get("root") or data_cfg.get("raw_dir", "data/raw")
    if str(root).replace("\\", "/").endswith("/gtsrb"):
        root = str(Path(root).parent)
    image_size = int(data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    validation_ratio = float(data_cfg.get("validation_ratio", 0.15))
    download = bool(data_cfg.get("download", True))

    train_aug_dataset = GTSRB(root=root, split="train", transform=build_transforms(image_size, True), download=download)
    train_eval_dataset = GTSRB(root=root, split="train", transform=build_transforms(image_size, False), download=False)
    test_dataset = GTSRB(root=root, split="test", transform=build_transforms(image_size, False), download=download)

    labels = [int(label) for _, label in train_aug_dataset._samples]
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_subset = Subset(train_aug_dataset, train_idx.tolist())
    val_subset = Subset(train_eval_dataset, val_idx.tolist())

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_subset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, test_dataset


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = config["training"].get("scheduler", "none").lower()
    epochs = int(config["training"]["epochs"])
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name in {"none", "null"}:
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float]:
    model.train()
    losses = []
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())

    return float(np.mean(losses)), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[float, float, list[int], list[int], list[np.ndarray]]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    probabilities = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)

        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        probabilities.extend(probs.cpu().numpy())

    return float(np.mean(losses)), accuracy_score(y_true, y_pred), y_true, y_pred, probabilities


def plot_training_curves(log_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="Validation")
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_acc"], label="Train")
    ax.plot(log_df["epoch"], log_df["val_acc"], label="Validation")
    ax.set_title("Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "accuracy_curve.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], figure_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(43)))
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True, square=True, xticklabels=range(43), yticklabels=range(43))
    ax.set_title("Confusion Matrix on Test Set")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_per_class_accuracy(y_true: list[int], y_pred: list[int], metrics_dir: Path, figure_dir: Path) -> None:
    rows = []
    for class_id in range(43):
        mask = np.array(y_true) == class_id
        class_total = int(mask.sum())
        class_correct = int((np.array(y_pred)[mask] == class_id).sum()) if class_total else 0
        rows.append(
            {
                "class_id": class_id,
                "class_name": GTSRB_LABELS[class_id],
                "total": class_total,
                "correct": class_correct,
                "accuracy": class_correct / class_total if class_total else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(df["class_id"], df["accuracy"])
    ax.set_title("Per-Class Accuracy on Test Set")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(df["class_id"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_class_accuracy.png", dpi=200)
    plt.close(fig)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.3337, 0.3064, 0.3171]).view(3, 1, 1)
    std = torch.tensor([0.2672, 0.2564, 0.2629]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


@torch.no_grad()
def collect_sample_predictions(
    model: nn.Module,
    dataset: GTSRB,
    device: torch.device,
    max_each: int = 24,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    correct_cases = []
    wrong_cases = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        pred = int(probs.argmax().item())
        case = {
            "image": denormalize(image),
            "label": int(label),
            "pred": pred,
            "confidence": float(probs[pred].item()),
        }
        if pred == int(label) and len(correct_cases) < max_each:
            correct_cases.append(case)
        if pred != int(label) and len(wrong_cases) < max_each:
            wrong_cases.append(case)
        if len(correct_cases) >= max_each and len(wrong_cases) >= max_each:
            break
    return correct_cases, wrong_cases


def tensor_to_pil(tensor: torch.Tensor, size: int = 96) -> Image.Image:
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def save_prediction_grid(cases: list[dict], output_path: Path, title: str, image_size: int = 96) -> None:
    if not cases:
        return
    cols = min(6, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    cell_w = 178
    cell_h = image_size + 66
    title_h = 48
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 15), title, fill="black", font=font)

    for idx, case in enumerate(cases):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = title_h + row * cell_h
        img = tensor_to_pil(case["image"], size=image_size)
        img_x = x + (cell_w - image_size) // 2
        canvas.paste(img, (img_x, y))
        draw.rectangle((img_x, y, img_x + image_size - 1, y + image_size - 1), outline=(220, 220, 220))
        label = case["label"]
        pred = case["pred"]
        draw.text((x + 8, y + image_size + 6), f"T {label}: {GTSRB_LABELS[label][:18]}", fill="black", font=font)
        color = "green" if label == pred else "red"
        draw.text((x + 8, y + image_size + 24), f"P {pred}: {GTSRB_LABELS[pred][:18]}", fill=color, font=font)
        draw.text((x + 8, y + image_size + 42), f"conf={case['confidence']:.3f}", fill="black", font=font)
    canvas.save(output_path)


def copy_report_assets(output_dirs: dict[str, Path], report_figure_dir: Path, report_table_dir: Path) -> None:
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure_names = {
        "loss_curve.png": "fig_05_baseline_loss_curve.png",
        "accuracy_curve.png": "fig_06_baseline_accuracy_curve.png",
        "confusion_matrix.png": "fig_07_baseline_confusion_matrix.png",
        "per_class_accuracy.png": "fig_08_baseline_per_class_accuracy.png",
    }
    for src_name, dst_name in figure_names.items():
        src = output_dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)

    sample_names = {
        "correct_samples_grid.png": "fig_09_baseline_correct_samples.png",
        "wrong_samples_grid.png": "fig_10_baseline_wrong_samples.png",
    }
    for src_name, dst_name in sample_names.items():
        src = output_dirs["samples"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)

    table_names = {
        "test_metrics.json": "table_03_baseline_test_metrics.json",
        "per_class_metrics.csv": "table_04_baseline_per_class_metrics.csv",
        "train_log.csv": "table_05_baseline_train_log.csv",
    }
    for src_name, dst_name in table_names.items():
        src = output_dirs["metrics"] / src_name
        if src.exists():
            shutil.copy2(src, report_table_dir / dst_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GTSRB classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_resnet18.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    result_dir = args.output_dir or Path(config["outputs"]["result_dir"])
    output_dirs = ensure_output_dirs(result_dir)
    shutil.copy2(args.config, output_dirs["root"] / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(
        output_dirs["root"] / "run_info.json",
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

    train_loader, val_loader, test_loader, test_dataset = make_loaders(config, seed)
    model = build_model(
        config["model"]["name"],
        num_classes=int(config["data"]["num_classes"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = -1.0
    history = []
    start_time = time.time()
    epochs = int(config["training"]["epochs"])

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, desc="val")
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                output_dirs["checkpoints"] / "best_model.pth",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_acc": best_val_acc,
            "epoch": epochs,
        },
        output_dirs["checkpoints"] / "last_model.pth",
    )

    log_df = pd.DataFrame(history)
    log_df.to_csv(output_dirs["metrics"] / "train_log.csv", index=False, encoding="utf-8-sig")
    plot_training_curves(log_df, output_dirs["figures"])

    best_checkpoint = torch.load(output_dirs["checkpoints"] / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred, probabilities = evaluate(model, test_loader, criterion, device, desc="test")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    test_metrics = {
        "model": config["model"]["name"],
        "parameter_count": count_parameters(model),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_accuracy": float(best_checkpoint["best_val_acc"]),
        "test_loss": float(test_loss),
        "accuracy": float(test_acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "training_seconds": float(time.time() - start_time),
    }
    write_json(output_dirs["metrics"] / "test_metrics.json", test_metrics)

    plot_confusion_matrix(y_true, y_pred, output_dirs["figures"])
    plot_per_class_accuracy(y_true, y_pred, output_dirs["metrics"], output_dirs["figures"])

    correct_cases, wrong_cases = collect_sample_predictions(model, test_dataset, device)
    save_prediction_grid(correct_cases, output_dirs["samples"] / "correct_samples_grid.png", "Correct Test Samples")
    save_prediction_grid(wrong_cases, output_dirs["samples"] / "wrong_samples_grid.png", "Wrong Test Samples")

    report_figure_dir = Path(config["outputs"].get("report_figure_dir", "reports/figures"))
    report_table_dir = Path(config["outputs"].get("report_table_dir", "reports/tables"))
    copy_report_assets(output_dirs, report_figure_dir, report_table_dir)

    with (output_dirs["logs"] / "train.log").open("w", encoding="utf-8") as f:
        f.write(json.dumps(test_metrics, ensure_ascii=False, indent=2))
        f.write("\n")

    print(f"Training completed: {result_dir}")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
