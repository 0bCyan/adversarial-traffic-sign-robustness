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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from src.attacks.methods import fgsm_attack, pgd_attack
from src.data.gtsrb_labels import GTSRB_LABELS
from src.evaluate_attacks import load_model
from src.evaluate_input_defense import preprocess_pixels

NORMALIZE_MEAN = torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1)
NORMALIZE_STD = torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dirs(result_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": result_dir,
        "metrics": result_dir / "metrics",
        "figures": result_dir / "figures",
        "samples": result_dir / "samples",
        "logs": result_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, check=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def make_loader(args) -> DataLoader:
    dataset = GTSRB(root=args.data_root, split="test", transform=build_transform(args.image_size), download=False)
    if args.indices.exists():
        indices = pd.read_csv(args.indices)["index"].astype(int).tolist()
        if args.max_samples and args.max_samples < len(indices):
            indices = indices[: args.max_samples]
        dataset = Subset(dataset, indices)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def clamp_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = NORMALIZE_MEAN.to(device)
    std = NORMALIZE_STD.to(device)
    return (0.0 - mean) / std, (1.0 - mean) / std


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images - mean) / std


@torch.no_grad()
def predict_full(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    top2 = probs.topk(k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return pred, conf, margin


def random_linf_noise(images: torch.Tensor, epsilon: float, data_min: torch.Tensor, data_max: torch.Tensor) -> torch.Tensor:
    noisy = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    return torch.max(torch.min(noisy, data_max), data_min).detach()


def tensor_to_pil(tensor: torch.Tensor, size: int = 122) -> Image.Image:
    arr = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def amplified_delta_to_pil(clean: torch.Tensor, adv: torch.Tensor, size: int = 122, scale: float = 30.0) -> Image.Image:
    delta = (adv - clean).detach().cpu()
    visible = (0.5 + delta * scale).clamp(0, 1)
    arr = (visible.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def save_random_vs_pgd_grid(cases: list[dict], output_path: Path) -> None:
    if not cases:
        return
    cols = 2
    rows = int(np.ceil(len(cases) / cols))
    image_size = 122
    cell_w = 610
    cell_h = image_size + 98
    title_h = 42
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 14), "Random noise control vs gradient PGD attack (epsilon=0.03)", fill="black", font=font)
    for idx, case in enumerate(cases):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = title_h + row * cell_h
        clean = case["clean"]
        random_img = case["random"]
        pgd_img = case["pgd"]
        canvas.paste(tensor_to_pil(clean, image_size), (x + 4, y))
        canvas.paste(tensor_to_pil(random_img, image_size), (x + 150, y))
        canvas.paste(tensor_to_pil(pgd_img, image_size), (x + 296, y))
        canvas.paste(amplified_delta_to_pil(clean, pgd_img, image_size), (x + 442, y))
        draw.text((x + 42, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 174, y + image_size + 5), "Random", fill="black", font=font)
        draw.text((x + 334, y + image_size + 5), "PGD", fill="red", font=font)
        draw.text((x + 464, y + image_size + 5), "PGD delta x30", fill="black", font=font)
        label = case["label"]
        draw.text((x + 4, y + image_size + 26), f"True {label}: {GTSRB_LABELS[label][:32]}", fill="black", font=font)
        draw.text((x + 4, y + image_size + 44), f"Clean pred {case['clean_pred']} conf={case['clean_conf']:.3f}", fill="green", font=font)
        draw.text((x + 4, y + image_size + 62), f"Random pred {case['random_pred']} conf={case['random_conf']:.3f}", fill="black", font=font)
        draw.text((x + 4, y + image_size + 80), f"PGD pred {case['pgd_pred']} conf={case['pgd_conf']:.3f}", fill="red", font=font)
    canvas.save(output_path)


def evaluate_random_noise_and_margins(model, loader, device, epsilons, dirs):
    data_min, data_max = clamp_bounds(device)
    rows = []
    margin_records = {name: [] for name in ["clean", "random_eps_0.03", "fgsm_eps_0.03", "pgd_eps_0.03"]}
    visual_cases = []

    for epsilon in epsilons:
        total = 0
        clean_correct = 0
        random_correct = 0
        attackable_changed = 0
        conf_drops = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_pred, clean_conf, clean_margin = predict_full(model, images)
            noisy = random_linf_noise(images, epsilon, data_min, data_max)
            random_pred, random_conf, random_margin = predict_full(model, noisy)

            if np.isclose(epsilon, 0.03):
                margin_records["clean"].extend(clean_margin.detach().cpu().numpy().tolist())
                margin_records["random_eps_0.03"].extend(random_margin.detach().cpu().numpy().tolist())

            total += int(labels.numel())
            clean_mask = clean_pred == labels
            clean_correct += int(clean_mask.sum().item())
            random_correct += int((random_pred == labels).sum().item())
            attackable_changed += int((clean_mask & (random_pred != labels)).sum().item())
            conf_drops.extend((clean_conf - random_conf).detach().cpu().numpy().tolist())

        rows.append(
            {
                "attack": "random_uniform_linf",
                "epsilon": epsilon,
                "clean_accuracy": clean_correct / total,
                "perturbed_accuracy": random_correct / total,
                "success_rate_on_clean_correct": attackable_changed / clean_correct if clean_correct else 0.0,
                "mean_confidence_drop": float(np.mean(conf_drops)),
                "total_images": total,
            }
        )

    # Compute FGSM/PGD margin distributions and visual cases once at epsilon=0.03.
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_pred, clean_conf, _ = predict_full(model, images)
        fgsm = fgsm_attack(model, images, labels, 0.03, data_min, data_max)
        pgd = pgd_attack(model, images, labels, 0.03, 0.005, 7, data_min, data_max)
        random_img = random_linf_noise(images, 0.03, data_min, data_max)
        fgsm_pred, _, fgsm_margin = predict_full(model, fgsm)
        pgd_pred, pgd_conf, pgd_margin = predict_full(model, pgd)
        random_pred, random_conf, _ = predict_full(model, random_img)
        margin_records["fgsm_eps_0.03"].extend(fgsm_margin.detach().cpu().numpy().tolist())
        margin_records["pgd_eps_0.03"].extend(pgd_margin.detach().cpu().numpy().tolist())
        total += int(labels.numel())

        clean_rgb = denormalize(images)
        random_rgb = denormalize(random_img)
        pgd_rgb = denormalize(pgd)
        success_mask = (clean_pred == labels) & (pgd_pred != labels)
        if len(visual_cases) < 6:
            for idx in torch.where(success_mask)[0].detach().cpu().numpy().tolist():
                if len(visual_cases) >= 6:
                    break
                visual_cases.append(
                    {
                        "clean": clean_rgb[idx].detach().cpu(),
                        "random": random_rgb[idx].detach().cpu(),
                        "pgd": pgd_rgb[idx].detach().cpu(),
                        "label": int(labels[idx].detach().cpu()),
                        "clean_pred": int(clean_pred[idx].detach().cpu()),
                        "random_pred": int(random_pred[idx].detach().cpu()),
                        "pgd_pred": int(pgd_pred[idx].detach().cpu()),
                        "clean_conf": float(clean_conf[idx].detach().cpu()),
                        "random_conf": float(random_conf[idx].detach().cpu()),
                        "pgd_conf": float(pgd_conf[idx].detach().cpu()),
                    }
                )

    margin_rows = []
    for name, values in margin_records.items():
        arr = np.asarray(values, dtype=np.float64)
        margin_rows.append(
            {
                "condition": name,
                "mean_top1_top2_margin": float(arr.mean()),
                "median_top1_top2_margin": float(np.median(arr)),
                "low_margin_rate_lt_0.20": float((arr < 0.20).mean()),
                "sample_count": int(arr.size),
            }
        )

    random_df = pd.DataFrame(rows)
    margin_df = pd.DataFrame(margin_rows)
    random_df.to_csv(dirs["metrics"] / "random_noise_control.csv", index=False, encoding="utf-8-sig")
    margin_df.to_csv(dirs["metrics"] / "margin_shift_metrics.csv", index=False, encoding="utf-8-sig")
    save_random_vs_pgd_grid(visual_cases, dirs["samples"] / "random_vs_pgd_eps_0.03_grid.png")
    return random_df, margin_df, margin_records


def evaluate_pgd_steps(model, loader, device, steps_list, epsilon, alpha):
    data_min, data_max = clamp_bounds(device)
    rows = []
    for steps in steps_list:
        total = 0
        clean_correct = 0
        adv_correct = 0
        changed = 0
        conf_drops = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_pred, clean_conf, _ = predict_full(model, images)
            adversarial = pgd_attack(model, images, labels, epsilon, alpha, int(steps), data_min, data_max)
            adv_pred, adv_conf, _ = predict_full(model, adversarial)
            total += int(labels.numel())
            clean_mask = clean_pred == labels
            clean_correct += int(clean_mask.sum().item())
            adv_correct += int((adv_pred == labels).sum().item())
            changed += int((clean_mask & (adv_pred != labels)).sum().item())
            conf_drops.extend((clean_conf - adv_conf).detach().cpu().numpy().tolist())
        rows.append(
            {
                "epsilon": epsilon,
                "alpha": alpha,
                "pgd_steps": int(steps),
                "clean_accuracy": clean_correct / total,
                "adversarial_accuracy": adv_correct / total,
                "attack_success_rate": changed / clean_correct if clean_correct else 0.0,
                "mean_confidence_drop": float(np.mean(conf_drops)),
                "total_images": total,
            }
        )
    return pd.DataFrame(rows)


def defense_variants() -> list[tuple[str, str, dict]]:
    return [
        ("none", "none", {}),
        ("gaussian_sigma_0.3", "gaussian_blur", {"kernel_size": 3, "sigma": 0.3}),
        ("gaussian_sigma_0.6", "gaussian_blur", {"kernel_size": 3, "sigma": 0.6}),
        ("gaussian_sigma_1.0", "gaussian_blur", {"kernel_size": 3, "sigma": 1.0}),
        ("median_k3", "median_filter", {"kernel_size": 3}),
        ("median_k5", "median_filter", {"kernel_size": 5}),
        ("jpeg_q95", "jpeg_compression", {"quality": 95}),
        ("jpeg_q75", "jpeg_compression", {"quality": 75}),
        ("jpeg_q50", "jpeg_compression", {"quality": 50}),
    ]


def apply_variant(images_rgb: torch.Tensor, method: str, params: dict) -> torch.Tensor:
    if method == "none":
        return images_rgb
    return preprocess_pixels(images_rgb, method, params)


def evaluate_defense_sweep(model, loader, device):
    data_min, data_max = clamp_bounds(device)
    variants = defense_variants()
    totals = {
        name: {"clean_correct": 0, "adv_correct": 0, "def_conf": []}
        for name, _, _ in variants
    }
    total = 0
    clean_base_correct = 0
    pgd_base_correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_pred, _, _ = predict_full(model, images)
        adversarial = pgd_attack(model, images, labels, 0.03, 0.005, 7, data_min, data_max)
        pgd_pred, _, _ = predict_full(model, adversarial)
        clean_rgb = denormalize(images)
        adv_rgb = denormalize(adversarial)

        total += int(labels.numel())
        clean_base_correct += int((clean_pred == labels).sum().item())
        pgd_base_correct += int((pgd_pred == labels).sum().item())

        for name, method, params in variants:
            clean_transformed = normalize(apply_variant(clean_rgb, method, params))
            adv_transformed = normalize(apply_variant(adv_rgb, method, params))
            clean_t_pred, _, _ = predict_full(model, clean_transformed)
            adv_t_pred, adv_t_conf, _ = predict_full(model, adv_transformed)
            totals[name]["clean_correct"] += int((clean_t_pred == labels).sum().item())
            totals[name]["adv_correct"] += int((adv_t_pred == labels).sum().item())
            totals[name]["def_conf"].extend(adv_t_conf.detach().cpu().numpy().tolist())

    clean_base_accuracy = clean_base_correct / total
    pgd_base_accuracy = pgd_base_correct / total
    rows = []
    for name, method, params in variants:
        row = {
            "variant": name,
            "method": method,
            "params": json.dumps(params, ensure_ascii=False),
            "clean_base_accuracy": clean_base_accuracy,
            "pgd_base_accuracy": pgd_base_accuracy,
            "clean_transformed_accuracy": totals[name]["clean_correct"] / total,
            "pgd_defended_accuracy": totals[name]["adv_correct"] / total,
            "clean_accuracy_drop": clean_base_accuracy - (totals[name]["clean_correct"] / total),
            "pgd_accuracy_gain": (totals[name]["adv_correct"] / total) - pgd_base_accuracy,
            "mean_defended_confidence": float(np.mean(totals[name]["def_conf"])),
            "total_images": total,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_random_vs_adversarial(random_df, attack_df, figure_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(random_df["epsilon"], random_df["perturbed_accuracy"], marker="o", label="Random uniform Linf")
    for attack in ["fgsm", "pgd"]:
        sub = attack_df[attack_df["attack"] == attack]
        ax.plot(sub["epsilon"], sub["adversarial_accuracy"], marker="o", label=attack.upper())
    ax.set_title("Random Noise Control vs Gradient Attacks")
    ax.set_xlabel("Epsilon in normalized space")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "random_vs_adversarial_accuracy.png", dpi=220)
    plt.close(fig)


def plot_pgd_steps(step_df, figure_dir):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(step_df["pgd_steps"], step_df["adversarial_accuracy"], marker="o", color="#1f77b4", label="Adv. accuracy")
    ax1.set_xlabel("PGD steps")
    ax1.set_ylabel("Adversarial accuracy")
    ax1.set_ylim(0.45, 0.90)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(step_df["pgd_steps"], step_df["attack_success_rate"], marker="s", color="#d62728", label="Attack success")
    ax2.set_ylabel("Attack success rate")
    ax2.set_ylim(0.10, 0.55)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    ax1.set_title("PGD Step Ablation at Epsilon=0.03")
    fig.tight_layout()
    fig.savefig(figure_dir / "pgd_step_ablation.png", dpi=220)
    plt.close(fig)


def plot_defense_sweep(sweep_df, figure_dir):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plot_df = sweep_df[sweep_df["variant"] != "none"].copy()
    ax.scatter(plot_df["clean_transformed_accuracy"], plot_df["pgd_defended_accuracy"], s=65)
    offsets = {
        "gaussian_sigma_0.3": (-92, -8),
        "gaussian_sigma_0.6": (8, -6),
        "gaussian_sigma_1.0": (8, 4),
        "median_k3": (8, 5),
        "median_k5": (8, 5),
        "jpeg_q95": (8, 6),
        "jpeg_q75": (8, 6),
        "jpeg_q50": (8, 6),
    }
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["variant"],
            (row["clean_transformed_accuracy"], row["pgd_defended_accuracy"]),
            fontsize=8,
            xytext=offsets.get(row["variant"], (5, 5)),
            textcoords="offset points",
        )
    ax.set_title("Defense Parameter Sweep: Clean-Robust Trade-off")
    ax.set_xlabel("Clean accuracy after preprocessing")
    ax.set_ylabel("PGD eps=0.03 defended accuracy")
    ax.set_xlim(0.961, 0.981)
    ax.set_ylim(0.595, 0.84)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "defense_parameter_tradeoff.png", dpi=220)
    plt.close(fig)

    ordered = sweep_df.sort_values("pgd_defended_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["variant"], ordered["pgd_defended_accuracy"])
    ax.axhline(float(sweep_df.iloc[0]["pgd_base_accuracy"]), linestyle="--", color="#d62728", label="Before defense")
    ax.set_title("PGD eps=0.03 Accuracy after Input Defense Variants")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 0.86)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "defense_parameter_bar.png", dpi=220)
    plt.close(fig)


def plot_margin_histogram(margin_records, figure_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 35)
    labels = {
        "clean": "Clean",
        "random_eps_0.03": "Random eps=0.03",
        "fgsm_eps_0.03": "FGSM eps=0.03",
        "pgd_eps_0.03": "PGD eps=0.03",
    }
    for key, values in margin_records.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, density=True, label=labels[key])
    ax.set_title("Top-1 vs Top-2 Probability Margin Shift")
    ax.set_xlabel("Top-1 probability - Top-2 probability")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "margin_shift_histogram.png", dpi=220)
    plt.close(fig)


def copy_report_assets(dirs):
    report_fig = Path("reports/figures")
    report_tbl = Path("reports/tables")
    report_fig.mkdir(parents=True, exist_ok=True)
    report_tbl.mkdir(parents=True, exist_ok=True)
    table_map = {
        "random_noise_control.csv": "table_09_random_noise_control.csv",
        "pgd_step_ablation.csv": "table_10_pgd_step_ablation.csv",
        "defense_parameter_sweep.csv": "table_11_defense_parameter_sweep.csv",
        "margin_shift_metrics.csv": "table_12_margin_shift_metrics.csv",
    }
    for src_name, dst_name in table_map.items():
        src = dirs["metrics"] / src_name
        if src.exists():
            shutil.copy2(src, report_tbl / dst_name)
    figure_map = {
        "random_vs_adversarial_accuracy.png": "fig_27_random_vs_adversarial_accuracy.png",
        "pgd_step_ablation.png": "fig_28_pgd_step_ablation.png",
        "defense_parameter_tradeoff.png": "fig_29_defense_parameter_tradeoff.png",
        "defense_parameter_bar.png": "fig_30_defense_parameter_bar.png",
        "margin_shift_histogram.png": "fig_31_margin_shift_histogram.png",
    }
    for src_name, dst_name in figure_map.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_fig / dst_name)
    sample = dirs["samples"] / "random_vs_pgd_eps_0.03_grid.png"
    if sample.exists():
        shutil.copy2(sample, report_fig / "fig_32_random_vs_pgd_visual_grid.png")


def main():
    parser = argparse.ArgumentParser(description="Run supplementary robustness experiments without model retraining.")
    parser.add_argument("--checkpoint", type=Path, default=Path("results/01_baseline/resnet18/checkpoints/best_model.pth"))
    parser.add_argument("--indices", type=Path, default=Path("results/02_attack/fgsm_pgd/metrics/selected_eval_indices.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/04_extended_analysis/supplementary_robustness"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=3000)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dirs = ensure_dirs(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(
        dirs["root"] / "run_info.json",
        {
            "experiment_name": "04_extended_supplementary_robustness",
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

    loader = make_loader(args)
    model = load_model(args.checkpoint, device)

    random_df, margin_df, margin_records = evaluate_random_noise_and_margins(
        model,
        loader,
        device,
        epsilons=[0.005, 0.01, 0.02, 0.03, 0.05],
        dirs=dirs,
    )
    step_df = evaluate_pgd_steps(model, loader, device, steps_list=[1, 2, 3, 5, 7, 10, 20], epsilon=0.03, alpha=0.005)
    sweep_df = evaluate_defense_sweep(model, loader, device)

    step_df.to_csv(dirs["metrics"] / "pgd_step_ablation.csv", index=False, encoding="utf-8-sig")
    sweep_df.to_csv(dirs["metrics"] / "defense_parameter_sweep.csv", index=False, encoding="utf-8-sig")

    attack_df = pd.read_csv("reports/tables/table_06_attack_metrics.csv")
    plot_random_vs_adversarial(random_df, attack_df, dirs["figures"])
    plot_pgd_steps(step_df, dirs["figures"])
    plot_defense_sweep(sweep_df, dirs["figures"])
    plot_margin_histogram(margin_records, dirs["figures"])
    copy_report_assets(dirs)

    with (dirs["logs"] / "extended_experiments.log").open("w", encoding="utf-8") as f:
        f.write("Random noise control\n")
        f.write(random_df.to_string(index=False))
        f.write("\n\nPGD step ablation\n")
        f.write(step_df.to_string(index=False))
        f.write("\n\nDefense parameter sweep\n")
        f.write(sweep_df.to_string(index=False))
        f.write("\n\nMargin metrics\n")
        f.write(margin_df.to_string(index=False))
        f.write("\n")
    print(f"Supplementary robustness experiments completed: {args.out_dir}")


if __name__ == "__main__":
    main()
