import argparse
import json
import platform
import shutil
import sys
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from tqdm import tqdm

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


def ensure_output_dirs(result_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": result_dir,
        "logs": result_dir / "logs",
        "metrics": result_dir / "metrics",
        "figures": result_dir / "figures",
        "samples": result_dir / "samples",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


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


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    model = build_model(
        checkpoint_config["model"]["name"],
        num_classes=int(checkpoint_config["data"]["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def clamp_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = NORMALIZE_MEAN.to(device)
    std = NORMALIZE_STD.to(device)
    data_min = (0.0 - mean) / std
    data_max = (1.0 - mean) / std
    return data_min, data_max


def denormalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor, size: int = 112) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def noise_to_pil(clean: torch.Tensor, adversarial: torch.Tensor, size: int = 112) -> Image.Image:
    noise = (adversarial - clean).detach().cpu()
    noise = noise.abs().mean(dim=0)
    if float(noise.max()) > 0:
        noise = noise / noise.max()
    arr = (plt.get_cmap("magma")(noise.numpy())[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


@torch.no_grad()
def clean_predictions(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def evaluate_attack(
    model: torch.nn.Module,
    loader: DataLoader,
    attack_name: str,
    epsilon: float,
    device: torch.device,
    pgd_alpha: float,
    pgd_steps: int,
    max_cases: int,
) -> tuple[dict, list[dict]]:
    data_min, data_max = clamp_bounds(device)
    total = 0
    clean_correct = 0
    adv_correct = 0
    clean_correct_attacked_wrong = 0
    confidence_drops = []
    cases = []

    for images, labels in tqdm(loader, desc=f"{attack_name} eps={epsilon}", leave=False, disable=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_pred, clean_conf = clean_predictions(model, images)

        if attack_name == "fgsm":
            adversarial = fgsm_attack(model, images, labels, epsilon, data_min, data_max)
        elif attack_name == "pgd":
            adversarial = pgd_attack(model, images, labels, epsilon, pgd_alpha, pgd_steps, data_min, data_max)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        adv_pred, adv_conf = clean_predictions(model, adversarial)
        total += int(labels.numel())
        clean_correct_mask = clean_pred == labels
        adv_correct_mask = adv_pred == labels
        clean_correct += int(clean_correct_mask.sum().item())
        adv_correct += int(adv_correct_mask.sum().item())
        attack_success_mask = clean_correct_mask & (adv_pred != labels)
        clean_correct_attacked_wrong += int(attack_success_mask.sum().item())
        confidence_drops.extend((clean_conf - adv_conf).detach().cpu().numpy().tolist())

        if len(cases) < max_cases:
            clean_rgb = denormalize_batch(images)
            adv_rgb = denormalize_batch(adversarial)
            for idx in torch.where(attack_success_mask)[0].detach().cpu().numpy().tolist():
                if len(cases) >= max_cases:
                    break
                cases.append(
                    {
                        "clean": clean_rgb[idx].detach().cpu(),
                        "adversarial": adv_rgb[idx].detach().cpu(),
                        "label": int(labels[idx].detach().cpu()),
                        "clean_pred": int(clean_pred[idx].detach().cpu()),
                        "adv_pred": int(adv_pred[idx].detach().cpu()),
                        "clean_conf": float(clean_conf[idx].detach().cpu()),
                        "adv_conf": float(adv_conf[idx].detach().cpu()),
                        "epsilon": epsilon,
                        "attack": attack_name,
                    }
                )

    clean_accuracy = clean_correct / total
    adv_accuracy = adv_correct / total
    attack_success_rate = clean_correct_attacked_wrong / clean_correct if clean_correct else 0.0
    metrics = {
        "attack": attack_name,
        "epsilon": epsilon,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "attack_success_rate": attack_success_rate,
        "mean_confidence_drop": float(np.mean(confidence_drops)),
        "total_images": total,
    }
    return metrics, cases


def save_triplet_grid(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        return
    cols = min(3, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    image_size = 112
    cell_w = 372
    cell_h = image_size + 88
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
        clean = tensor_to_pil(case["clean"], image_size)
        adv = tensor_to_pil(case["adversarial"], image_size)
        noise = noise_to_pil(case["clean"], case["adversarial"], image_size)

        canvas.paste(clean, (x + 4, y))
        canvas.paste(noise, (x + 128, y))
        canvas.paste(adv, (x + 252, y))
        draw.text((x + 38, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 164, y + image_size + 5), "Noise", fill="black", font=font)
        draw.text((x + 286, y + image_size + 5), "Adversarial", fill="black", font=font)

        label = case["label"]
        adv_pred = case["adv_pred"]
        draw.text((x + 4, y + image_size + 24), f"True: {label} {GTSRB_LABELS[label][:24]}", fill="black", font=font)
        draw.text(
            (x + 4, y + image_size + 42),
            f"Clean: {case['clean_pred']} conf={case['clean_conf']:.3f}",
            fill="green",
            font=font,
        )
        draw.text(
            (x + 4, y + image_size + 60),
            f"Adv: {adv_pred} conf={case['adv_conf']:.3f}",
            fill="red",
            font=font,
        )
    canvas.save(output_path)


def plot_attack_curves(metrics_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, group in metrics_df.groupby("attack"):
        ax.plot(group["epsilon"], group["adversarial_accuracy"], marker="o", label=attack.upper())
    ax.set_title("Adversarial Accuracy under Different Epsilon")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "epsilon_accuracy_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, group in metrics_df.groupby("attack"):
        ax.plot(group["epsilon"], group["attack_success_rate"], marker="o", label=attack.upper())
    ax.set_title("Attack Success Rate under Different Epsilon")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "attack_success_rate_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, group in metrics_df.groupby("attack"):
        ax.plot(group["epsilon"], group["mean_confidence_drop"], marker="o", label=attack.upper())
    ax.set_title("Mean Confidence Drop")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Clean confidence - adversarial confidence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "confidence_drop_curve.png", dpi=200)
    plt.close(fig)


def save_attack_outputs(metrics: list[dict], dirs: dict[str, Path]) -> None:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(dirs["metrics"] / "attack_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["metrics"] / "attack_summary.json", {"metrics": metrics})
    plot_attack_curves(metrics_df, dirs["figures"])


def copy_report_assets(dirs: dict[str, Path], report_figure_dir: Path, report_table_dir: Path) -> None:
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "epsilon_accuracy_curve.png": "fig_11_attack_accuracy_curve.png",
        "attack_success_rate_curve.png": "fig_12_attack_success_curve.png",
        "confidence_drop_curve.png": "fig_13_attack_confidence_drop.png",
    }
    for src_name, dst_name in figure_map.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)

    for src_name, dst_name in {
        "fgsm_eps_0.03_triplets.png": "fig_14_fgsm_eps003_triplets.png",
        "pgd_eps_0.03_triplets.png": "fig_15_pgd_eps003_triplets.png",
    }.items():
        src = dirs["samples"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)

    src = dirs["metrics"] / "attack_metrics.csv"
    if src.exists():
        shutil.copy2(src, report_table_dir / "table_06_attack_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FGSM and PGD attacks on a GTSRB classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/attack_fgsm_pgd.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))

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

    image_size = int(config["data"]["image_size"])
    batch_size = int(config["data"]["batch_size"])
    root = config["data"].get("root", "data/raw")
    dataset = GTSRB(root=root, split="test", transform=build_eval_transform(image_size), download=False)
    max_eval_samples = int(config["data"].get("max_eval_samples", 0))
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

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    model = load_model(Path(config["model"]["checkpoint"]), device)

    all_metrics = []
    selected_cases: dict[str, list[dict]] = {}
    max_cases = int(config["outputs"].get("max_cases_per_grid", 12))
    for attack_name in ["fgsm", "pgd"]:
        attack_cfg = config["attacks"][attack_name]
        for epsilon in attack_cfg["epsilons"]:
            pgd_alpha = float(config["attacks"].get("pgd", {}).get("alpha", 0.005))
            pgd_steps = int(config["attacks"].get("pgd", {}).get("steps", 10))
            metrics, cases = evaluate_attack(
                model,
                loader,
                attack_name,
                float(epsilon),
                device,
                pgd_alpha,
                pgd_steps,
                max_cases,
            )
            all_metrics.append(metrics)
            selected_cases[f"{attack_name}_{epsilon}"] = cases
            print(json.dumps(metrics, ensure_ascii=False))
            save_attack_outputs(all_metrics, dirs)

            eps_text = f"{float(epsilon):.2f}".replace(".", ".")
            if float(epsilon) in {0.01, 0.03, 0.05}:
                save_triplet_grid(
                    cases,
                    dirs["samples"] / f"{attack_name}_eps_{eps_text}_triplets.png",
                    f"{attack_name.upper()} Attack Examples, epsilon={epsilon}",
                )

    metrics_df = pd.DataFrame(all_metrics)
    save_attack_outputs(all_metrics, dirs)

    copy_report_assets(
        dirs,
        Path(config["outputs"].get("report_figure_dir", "reports/figures")),
        Path(config["outputs"].get("report_table_dir", "reports/tables")),
    )
    with (dirs["logs"] / "attack.log").open("w", encoding="utf-8") as f:
        f.write(metrics_df.to_string(index=False))
        f.write("\n")
    print(f"Attack evaluation completed: {result_dir}")


if __name__ == "__main__":
    main()
