import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.attacks.methods import pgd_attack
from src.data.gtsrb_labels import GTSRB_LABELS
from src.defenses.preprocessing import preprocess_pixels
from src.experiment_utils import (
    clamp_bounds,
    copy_config,
    denormalize,
    ensure_output_dirs,
    load_checkpoint_model,
    load_config,
    make_loader,
    make_test_dataset,
    normalize,
    predict,
    tensor_to_pil,
    write_json,
    write_run_info,
)


def bpda_jpeg_pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    quality: int,
    data_min: torch.Tensor,
    data_max: torch.Tensor,
) -> torch.Tensor:
    original = images.detach()
    adversarial = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    adversarial = torch.max(torch.min(adversarial, data_max), data_min).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)
        defended_rgb = preprocess_pixels(denormalize(adversarial), "jpeg_compression", {"quality": quality})
        defended_norm = normalize(defended_rgb)
        bpda_input = adversarial + (defended_norm - adversarial).detach()
        logits = model(bpda_input)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adversarial = adversarial + alpha * adversarial.grad.sign()
        delta = torch.clamp(adversarial - original, min=-epsilon, max=epsilon)
        adversarial = torch.max(torch.min(original + delta, data_max), data_min).detach()
    return adversarial


def save_adaptive_grid(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        return
    cols = min(3, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    image_size = 112
    cell_w = 420
    cell_h = image_size + 96
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
        canvas.paste(tensor_to_pil(case["clean"], image_size), (x + 4, y))
        canvas.paste(tensor_to_pil(case["standard_defended"], image_size), (x + 146, y))
        canvas.paste(tensor_to_pil(case["adaptive_defended"], image_size), (x + 288, y))
        draw.text((x + 42, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 158, y + image_size + 5), "PGD+JPEG", fill="green", font=font)
        draw.text((x + 296, y + image_size + 5), "BPDA+JPEG", fill="red", font=font)
        label = case["label"]
        draw.text((x + 4, y + image_size + 25), f"True: {label} {GTSRB_LABELS[label][:24]}", fill="black", font=font)
        draw.text((x + 4, y + image_size + 44), f"Clean pred: {case['clean_pred']} conf={case['clean_conf']:.3f}", fill="black", font=font)
        draw.text((x + 4, y + image_size + 63), f"Std JPEG: {case['standard_def_pred']} conf={case['standard_def_conf']:.3f}", fill="green", font=font)
        draw.text((x + 4, y + image_size + 81), f"BPDA JPEG: {case['adaptive_def_pred']} conf={case['adaptive_def_conf']:.3f}", fill="red", font=font)
    canvas.save(output_path)


def plot_adaptive(metrics_df: pd.DataFrame, figure_dir: Path) -> None:
    row = metrics_df.iloc[0]
    labels = [
        "Clean",
        "Standard PGD",
        "Standard PGD + JPEG",
        "Adaptive BPDA + JPEG",
    ]
    values = [
        row["clean_accuracy"],
        row["standard_pgd_accuracy_before_jpeg"],
        row["standard_pgd_accuracy_after_jpeg"],
        row["adaptive_bpda_accuracy_after_jpeg"],
    ]
    colors = ["#4F8F6B", "#C64E4E", "#5E8CCB", "#9D4E9D"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(labels, values, color=colors)
    ax.set_title("JPEG Defense under Standard PGD vs Adaptive BPDA Attack")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "adaptive_jpeg_bpda_accuracy_bar.png", dpi=220)
    plt.close(fig)


def evaluate(config: dict, dirs: dict[str, Path]) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(Path(config["model"]["checkpoint"]), device)
    dataset = make_test_dataset(config, dirs)
    loader = make_loader(
        dataset,
        int(config["data"]["batch_size"]),
        int(config["data"].get("num_workers", 0)),
        shuffle=False,
    )
    data_min, data_max = clamp_bounds(device)
    attack_cfg = config["attack"]
    defense_cfg = config["defense"]
    epsilon = float(attack_cfg["epsilon"])
    alpha = float(attack_cfg.get("alpha", 0.005))
    steps = int(attack_cfg.get("steps", 7))
    quality = int(defense_cfg.get("quality", 75))

    total = 0
    clean_correct = 0
    standard_before_correct = 0
    standard_after_correct = 0
    adaptive_after_correct = 0
    standard_attack_success = 0
    adaptive_attack_success_after_jpeg = 0
    bpda_cases: list[dict] = []
    max_cases = int(config["outputs"].get("max_cases_per_grid", 9))

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_pred, clean_conf, _ = predict(model, images)
        clean_mask = clean_pred == labels

        standard_adv = pgd_attack(model, images, labels, epsilon, alpha, steps, data_min, data_max)
        standard_pred, _, _ = predict(model, standard_adv)
        standard_def_rgb = preprocess_pixels(denormalize(standard_adv), defense_cfg["method"], defense_cfg)
        standard_def_norm = normalize(standard_def_rgb)
        standard_def_pred, standard_def_conf, _ = predict(model, standard_def_norm)

        adaptive_adv = bpda_jpeg_pgd_attack(model, images, labels, epsilon, alpha, steps, quality, data_min, data_max)
        adaptive_def_rgb = preprocess_pixels(denormalize(adaptive_adv), defense_cfg["method"], defense_cfg)
        adaptive_def_norm = normalize(adaptive_def_rgb)
        adaptive_def_pred, adaptive_def_conf, _ = predict(model, adaptive_def_norm)

        total += int(labels.numel())
        standard_before_mask = standard_pred == labels
        standard_after_mask = standard_def_pred == labels
        adaptive_after_mask = adaptive_def_pred == labels
        clean_correct += int(clean_mask.sum().item())
        standard_before_correct += int(standard_before_mask.sum().item())
        standard_after_correct += int(standard_after_mask.sum().item())
        adaptive_after_correct += int(adaptive_after_mask.sum().item())
        standard_attack_success += int((clean_mask & ~standard_before_mask).sum().item())
        adaptive_attack_success_after_jpeg += int((clean_mask & ~adaptive_after_mask).sum().item())

        useful_case_mask = clean_mask & standard_after_mask & ~adaptive_after_mask
        if len(bpda_cases) < max_cases:
            clean_rgb = denormalize(images)
            for idx in torch.where(useful_case_mask)[0].detach().cpu().numpy().tolist():
                if len(bpda_cases) >= max_cases:
                    break
                bpda_cases.append(
                    {
                        "clean": clean_rgb[idx].detach().cpu(),
                        "standard_defended": standard_def_rgb[idx].detach().cpu(),
                        "adaptive_defended": adaptive_def_rgb[idx].detach().cpu(),
                        "label": int(labels[idx].detach().cpu()),
                        "clean_pred": int(clean_pred[idx].detach().cpu()),
                        "standard_def_pred": int(standard_def_pred[idx].detach().cpu()),
                        "adaptive_def_pred": int(adaptive_def_pred[idx].detach().cpu()),
                        "clean_conf": float(clean_conf[idx].detach().cpu()),
                        "standard_def_conf": float(standard_def_conf[idx].detach().cpu()),
                        "adaptive_def_conf": float(adaptive_def_conf[idx].detach().cpu()),
                    }
                )

    metrics = pd.DataFrame(
        [
            {
                "attack": "pgd_bpda_jpeg",
                "epsilon": epsilon,
                "pgd_alpha": alpha,
                "pgd_steps": steps,
                "jpeg_quality": quality,
                "total_images": total,
                "clean_accuracy": clean_correct / total,
                "standard_pgd_accuracy_before_jpeg": standard_before_correct / total,
                "standard_pgd_accuracy_after_jpeg": standard_after_correct / total,
                "adaptive_bpda_accuracy_after_jpeg": adaptive_after_correct / total,
                "standard_pgd_success_rate_before_jpeg_on_clean_correct": standard_attack_success / clean_correct if clean_correct else 0.0,
                "adaptive_bpda_success_rate_after_jpeg_on_clean_correct": adaptive_attack_success_after_jpeg / clean_correct if clean_correct else 0.0,
            }
        ]
    )
    plot_adaptive(metrics, dirs["figures"])
    save_adaptive_grid(
        bpda_cases,
        dirs["samples"] / "adaptive_jpeg_bpda_examples.png",
        "Adaptive BPDA attack against JPEG Q75 defense",
    )
    return metrics


def copy_report_assets(dirs: dict[str, Path], config: dict) -> None:
    report_figure_dir = Path(config["outputs"].get("report_figure_dir", "reports/figures"))
    report_table_dir = Path(config["outputs"].get("report_table_dir", "reports/tables"))
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure = dirs["figures"] / "adaptive_jpeg_bpda_accuracy_bar.png"
    if figure.exists():
        shutil.copy2(figure, report_figure_dir / "fig_45_adaptive_jpeg_bpda_accuracy.png")
    sample = dirs["samples"] / "adaptive_jpeg_bpda_examples.png"
    if sample.exists():
        shutil.copy2(sample, report_figure_dir / "fig_46_adaptive_jpeg_bpda_examples.png")
    table = dirs["metrics"] / "adaptive_jpeg_bpda_metrics.csv"
    if table.exists():
        shutil.copy2(table, report_table_dir / "table_18_adaptive_jpeg_bpda_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an adaptive BPDA-style attack against JPEG defense.")
    parser.add_argument("--config", type=Path, default=Path("configs/adaptive_jpeg_attack.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    result_dir = Path(config["outputs"]["result_dir"])
    dirs = ensure_output_dirs(result_dir)
    copy_config(args.config, dirs["root"])
    write_run_info(dirs, config, sys.argv)

    metrics = evaluate(config, dirs)
    metrics.to_csv(dirs["metrics"] / "adaptive_jpeg_bpda_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["metrics"] / "adaptive_jpeg_bpda_summary.json", {"metrics": metrics.to_dict(orient="records")})
    copy_report_assets(dirs, config)
    with (dirs["logs"] / "adaptive_jpeg_bpda.log").open("w", encoding="utf-8") as f:
        f.write(metrics.to_string(index=False))
        f.write("\n")
    print(f"Adaptive JPEG BPDA attack completed: {result_dir}")


if __name__ == "__main__":
    main()
