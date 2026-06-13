import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from src.attacks.methods import fgsm_attack, pgd_attack
from src.data.gtsrb_labels import GTSRB_LABELS
from src.defenses.preprocessing import jpeg_compress_batch
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


def make_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_name: str,
    epsilon: float,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    if attack_name == "clean":
        return images
    data_min, data_max = clamp_bounds(device)
    if attack_name == "fgsm":
        return fgsm_attack(model, images, labels, epsilon, data_min, data_max)
    if attack_name == "pgd":
        pgd_cfg = config["attacks"]["pgd"]
        return pgd_attack(
            model,
            images,
            labels,
            epsilon,
            float(pgd_cfg.get("alpha", 0.005)),
            int(pgd_cfg.get("steps", 7)),
            data_min,
            data_max,
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


def batch_psnr(reference: torch.Tensor, compared: torch.Tensor) -> list[float]:
    mse = (reference - compared).pow(2).flatten(1).mean(dim=1).detach().cpu().numpy()
    values = []
    for item in mse:
        values.append(float("inf") if item <= 1e-12 else float(10.0 * np.log10(1.0 / item)))
    return values


def save_quality_grid(cases: list[dict], output_path: Path, title: str, qualities: list[int]) -> None:
    if not cases:
        return
    image_size = 112
    cols = 2 + len(qualities)
    rows = len(cases)
    cell_w = 166
    cell_h = image_size + 74
    title_h = 52
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 15), title, fill="black", font=font)

    headers = ["Clean", "Attack"] + [f"JPEG Q{quality}" for quality in qualities]
    for col, header in enumerate(headers):
        draw.text((col * cell_w + 42, title_h - 18), header, fill="black", font=font)

    for row, case in enumerate(cases):
        y = title_h + row * cell_h
        images = [case["clean"], case["attacked"]] + [case[f"jpeg_{quality}"] for quality in qualities]
        for col, image_tensor in enumerate(images):
            x = col * cell_w + 27
            image = tensor_to_pil(image_tensor, size=image_size)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + image_size - 1, y + image_size - 1), outline=(220, 220, 220))

        label = case["label"]
        draw.text((8, y + image_size + 7), f"T {label}: {GTSRB_LABELS[label][:18]}", fill="black", font=font)
        draw.text((8, y + image_size + 25), f"Adv {case['adv_pred']} {case['adv_conf']:.3f}", fill="red", font=font)
        for col, quality in enumerate(qualities, start=2):
            pred = case[f"jpeg_{quality}_pred"]
            conf = case[f"jpeg_{quality}_conf"]
            color = "green" if pred == label else "black"
            draw.text((col * cell_w + 8, y + image_size + 25), f"P {pred} {conf:.3f}", fill=color, font=font)
    canvas.save(output_path)


def evaluate(config: dict, model: torch.nn.Module, loader, device: torch.device, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    qualities = [int(item) for item in config["defense"]["qualities"]]
    max_cases = int(config["outputs"].get("max_cases_per_grid", 9))

    attack_names = [name for name in ["clean", "fgsm", "pgd"] if name in config["attacks"]]
    for attack_name in attack_names:
        for epsilon in config["attacks"][attack_name]["epsilons"]:
            epsilon = float(epsilon)
            saved_cases = []
            total = 0
            clean_correct = 0
            adv_correct = 0
            attackable = 0
            quality_totals = {
                quality: {"correct": 0, "recovered": 0, "conf": [], "psnr": []}
                for quality in qualities
            }

            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                clean_pred, _, _ = predict(model, images)
                attacked = make_attack(model, images, labels, attack_name, epsilon, config, device)
                adv_pred, adv_conf, _ = predict(model, attacked)
                clean_rgb = denormalize(images)
                attacked_rgb = denormalize(attacked)

                total += int(labels.numel())
                clean_mask = clean_pred == labels
                adv_mask = adv_pred == labels
                attack_success = clean_mask & ~adv_mask
                clean_correct += int(clean_mask.sum().item())
                adv_correct += int(adv_mask.sum().item())
                attackable += int(attack_success.sum().item())

                batch_case: dict | None = None
                case_idx = None
                if attack_name != "clean" and np.isclose(epsilon, 0.03) and len(saved_cases) < max_cases:
                    candidate_indices = torch.where(attack_success)[0].detach().cpu().numpy().tolist()
                    if candidate_indices:
                        case_idx = candidate_indices[0]
                        batch_case = {
                            "clean": clean_rgb[case_idx].detach().cpu(),
                            "attacked": attacked_rgb[case_idx].detach().cpu(),
                            "label": int(labels[case_idx].detach().cpu()),
                            "adv_pred": int(adv_pred[case_idx].detach().cpu()),
                            "adv_conf": float(adv_conf[case_idx].detach().cpu()),
                        }

                for quality in qualities:
                    defended_rgb = jpeg_compress_batch(attacked_rgb, quality)
                    defended_norm = normalize(defended_rgb)
                    def_pred, def_conf, _ = predict(model, defended_norm)
                    def_mask = def_pred == labels
                    recovered_mask = attack_success & def_mask
                    quality_totals[quality]["correct"] += int(def_mask.sum().item())
                    quality_totals[quality]["recovered"] += int(recovered_mask.sum().item())
                    quality_totals[quality]["conf"].extend(def_conf.detach().cpu().numpy().tolist())
                    quality_totals[quality]["psnr"].extend(batch_psnr(attacked_rgb, defended_rgb))

                    if batch_case is not None and case_idx is not None:
                        batch_case[f"jpeg_{quality}"] = defended_rgb[case_idx].detach().cpu()
                        batch_case[f"jpeg_{quality}_pred"] = int(def_pred[case_idx].detach().cpu())
                        batch_case[f"jpeg_{quality}_conf"] = float(def_conf[case_idx].detach().cpu())

                if batch_case is not None:
                    saved_cases.append(batch_case)

            clean_accuracy = clean_correct / total
            attack_accuracy = adv_correct / total
            for quality, values in quality_totals.items():
                row = {
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "jpeg_quality": quality,
                    "clean_accuracy": clean_accuracy,
                    "accuracy_before_jpeg": attack_accuracy,
                    "accuracy_after_jpeg": values["correct"] / total,
                    "recovery_rate_on_successful_attacks": values["recovered"] / attackable if attackable else 0.0,
                    "mean_confidence_after_jpeg": float(np.mean(values["conf"])),
                    "mean_psnr_vs_attack": float(np.mean([v for v in values["psnr"] if np.isfinite(v)])),
                    "total_images": total,
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

            if attack_name != "clean" and np.isclose(epsilon, 0.03):
                save_quality_grid(
                    saved_cases[:max_cases],
                    dirs["samples"] / f"{attack_name}_eps_0.03_jpeg_quality_grid.png",
                    f"JPEG quality ablation, {attack_name.upper()} epsilon=0.03",
                    qualities,
                )

    return pd.DataFrame(rows)


def plot_results(metrics_df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    attack_df = metrics_df[metrics_df["attack"] != "clean"]
    for attack in attack_df["attack"].unique():
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = attack_df[attack_df["attack"] == attack]
        for quality, group in subset.groupby("jpeg_quality"):
            ax.plot(group["epsilon"], group["accuracy_after_jpeg"], marker="o", label=f"JPEG Q{quality}")
        before = subset.groupby("epsilon")["accuracy_before_jpeg"].first().reset_index()
        ax.plot(before["epsilon"], before["accuracy_before_jpeg"], marker="x", linestyle="--", label="before JPEG")
        ax.set_title(f"JPEG Quality Ablation under {attack.upper()}")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"{attack}_jpeg_quality_curve.png", dpi=200)
        plt.close(fig)

    eps_subset = attack_df[np.isclose(attack_df["epsilon"], 0.03)]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{row.attack.upper()} Q{row.jpeg_quality}" for row in eps_subset.itertuples()]
    values = eps_subset["accuracy_after_jpeg"].tolist()
    ax.bar(labels, values)
    ax.set_title("JPEG Quality Defense Accuracy at Epsilon=0.03")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "jpeg_quality_eps003_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    clean_df = metrics_df[metrics_df["attack"] == "clean"]
    if not clean_df.empty:
        ax.plot(clean_df["jpeg_quality"], clean_df["accuracy_after_jpeg"], marker="o", label="clean after JPEG")
        ax.axhline(clean_df["clean_accuracy"].iloc[0], linestyle="--", color="gray", label="clean baseline")
        ax.set_title("Clean Accuracy Cost of JPEG Compression")
        ax.set_xlabel("JPEG Quality")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "jpeg_clean_accuracy_cost.png", dpi=200)
    plt.close(fig)


def copy_report_assets(dirs: dict[str, Path], config: dict) -> None:
    report_figure_dir = Path(config["outputs"].get("report_figure_dir", "reports/figures"))
    report_table_dir = Path(config["outputs"].get("report_table_dir", "reports/tables"))
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "fgsm_jpeg_quality_curve.png": "fig_28_fgsm_jpeg_quality_curve.png",
        "pgd_jpeg_quality_curve.png": "fig_29_pgd_jpeg_quality_curve.png",
        "jpeg_quality_eps003_bar.png": "fig_30_jpeg_quality_eps003_bar.png",
        "jpeg_clean_accuracy_cost.png": "fig_31_jpeg_clean_accuracy_cost.png",
    }
    for src_name, dst_name in figure_map.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)
    for src_name, dst_name in {
        "fgsm_eps_0.03_jpeg_quality_grid.png": "fig_32_fgsm_jpeg_quality_examples.png",
        "pgd_eps_0.03_jpeg_quality_grid.png": "fig_33_pgd_jpeg_quality_examples.png",
    }.items():
        src = dirs["samples"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)
    metrics = dirs["metrics"] / "jpeg_quality_metrics.csv"
    if metrics.exists():
        shutil.copy2(metrics, report_table_dir / "table_12_jpeg_quality_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JPEG quality ablation as an input defense.")
    parser.add_argument("--config", type=Path, default=Path("configs/defense_jpeg_ablation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    result_dir = args.output_dir or Path(config["outputs"]["result_dir"])
    dirs = ensure_output_dirs(result_dir)
    copy_config(args.config, dirs["root"])
    write_run_info(dirs, config, sys.argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(Path(config["model"]["checkpoint"]), device)
    dataset = make_test_dataset(config, dirs)
    loader = make_loader(
        dataset,
        int(config["data"]["batch_size"]),
        int(config["data"].get("num_workers", 0)),
        shuffle=False,
    )

    metrics_df = evaluate(config, model, loader, device, dirs)
    metrics_df.to_csv(dirs["metrics"] / "jpeg_quality_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["metrics"] / "jpeg_quality_summary.json", {"metrics": metrics_df.to_dict(orient="records")})
    plot_results(metrics_df, dirs)
    copy_report_assets(dirs, config)
    with (dirs["logs"] / "jpeg_quality_ablation.log").open("w", encoding="utf-8") as f:
        f.write(metrics_df.to_string(index=False))
        f.write("\n")
    print(f"JPEG quality ablation completed: {result_dir}")


if __name__ == "__main__":
    main()
