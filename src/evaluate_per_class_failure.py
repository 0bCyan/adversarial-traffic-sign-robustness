import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


def _empty_class_stats() -> dict[str, float]:
    return {
        "total": 0,
        "clean_correct": 0,
        "adv_correct": 0,
        "def_correct": 0,
        "attack_success": 0,
        "recovered": 0,
        "still_failed": 0,
    }


def save_failure_grid(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        return
    cols = min(3, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    image_size = 112
    cell_w = 390
    cell_h = image_size + 98
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
        canvas.paste(tensor_to_pil(case["adversarial"], image_size), (x + 134, y))
        canvas.paste(tensor_to_pil(case["defended"], image_size), (x + 264, y))
        draw.text((x + 42, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 160, y + image_size + 5), "PGD", fill="red", font=font)
        draw.text((x + 282, y + image_size + 5), "JPEG Q75", fill="darkred", font=font)
        label = case["label"]
        draw.text((x + 4, y + image_size + 26), f"True: {label} {GTSRB_LABELS[label][:24]}", fill="black", font=font)
        draw.text((x + 4, y + image_size + 45), f"Clean: {case['clean_pred']} conf={case['clean_conf']:.3f}", fill="green", font=font)
        draw.text((x + 4, y + image_size + 64), f"PGD: {case['adv_pred']} conf={case['adv_conf']:.3f}", fill="red", font=font)
        draw.text((x + 4, y + image_size + 82), f"JPEG: {case['def_pred']} conf={case['def_conf']:.3f}", fill="darkred", font=font)
    canvas.save(output_path)


def plot_per_class(per_class_df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = per_class_df[per_class_df["clean_correct"] >= 5].copy()
    top_attack = plot_df.sort_values("attack_success_rate_on_clean_correct", ascending=False).head(12)
    labels = [f"{int(r.class_id)}\n{str(r.label_name)[:14]}" for _, r in top_attack.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, top_attack["attack_success_rate_on_clean_correct"], color="#C64E4E")
    ax.set_title("Most Vulnerable Classes under PGD epsilon=0.03")
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_class_attack_success_top.png", dpi=220)
    plt.close(fig)

    top_recovery = plot_df.sort_values("recovery_rate_on_successful_attacks", ascending=False).head(12)
    labels = [f"{int(r.class_id)}\n{str(r.label_name)[:14]}" for _, r in top_recovery.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, top_recovery["recovery_rate_on_successful_attacks"], color="#4F8F6B")
    ax.set_title("Classes Most Recovered by JPEG Q75 after PGD")
    ax.set_ylabel("Recovery rate on successful attacks")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_class_defense_recovery_top.png", dpi=220)
    plt.close(fig)


def evaluate(config: dict, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    class_stats = defaultdict(_empty_class_stats)
    failure_cases: list[dict] = []
    max_cases = int(config["outputs"].get("max_failure_cases", 9))

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_pred, clean_conf, _ = predict(model, images)
        adversarial = pgd_attack(
            model,
            images,
            labels,
            float(attack_cfg["epsilon"]),
            float(attack_cfg.get("alpha", 0.005)),
            int(attack_cfg.get("steps", 7)),
            data_min,
            data_max,
        )
        adv_pred, adv_conf, _ = predict(model, adversarial)
        adv_rgb = denormalize(adversarial)
        defended_rgb = preprocess_pixels(adv_rgb, defense_cfg["method"], defense_cfg)
        defended_norm = normalize(defended_rgb)
        def_pred, def_conf, _ = predict(model, defended_norm)
        clean_rgb = denormalize(images)

        clean_correct = clean_pred == labels
        adv_correct = adv_pred == labels
        def_correct = def_pred == labels
        attack_success = clean_correct & ~adv_correct
        recovered = attack_success & def_correct
        still_failed = attack_success & ~def_correct

        for idx in range(labels.numel()):
            label = int(labels[idx].detach().cpu())
            stats = class_stats[label]
            stats["total"] += 1
            stats["clean_correct"] += int(clean_correct[idx].item())
            stats["adv_correct"] += int(adv_correct[idx].item())
            stats["def_correct"] += int(def_correct[idx].item())
            stats["attack_success"] += int(attack_success[idx].item())
            stats["recovered"] += int(recovered[idx].item())
            stats["still_failed"] += int(still_failed[idx].item())

        if len(failure_cases) < max_cases:
            for idx in torch.where(still_failed)[0].detach().cpu().numpy().tolist():
                if len(failure_cases) >= max_cases:
                    break
                failure_cases.append(
                    {
                        "clean": clean_rgb[idx].detach().cpu(),
                        "adversarial": adv_rgb[idx].detach().cpu(),
                        "defended": defended_rgb[idx].detach().cpu(),
                        "label": int(labels[idx].detach().cpu()),
                        "clean_pred": int(clean_pred[idx].detach().cpu()),
                        "adv_pred": int(adv_pred[idx].detach().cpu()),
                        "def_pred": int(def_pred[idx].detach().cpu()),
                        "clean_conf": float(clean_conf[idx].detach().cpu()),
                        "adv_conf": float(adv_conf[idx].detach().cpu()),
                        "def_conf": float(def_conf[idx].detach().cpu()),
                    }
                )

    rows = []
    for class_id in range(len(GTSRB_LABELS)):
        stats = class_stats[class_id]
        total = stats["total"]
        clean_correct = stats["clean_correct"]
        attack_success = stats["attack_success"]
        rows.append(
            {
                "class_id": class_id,
                "label_name": GTSRB_LABELS[class_id],
                "total_images": total,
                "clean_correct": clean_correct,
                "clean_accuracy": stats["clean_correct"] / total if total else 0.0,
                "pgd_accuracy": stats["adv_correct"] / total if total else 0.0,
                "jpeg_defended_accuracy": stats["def_correct"] / total if total else 0.0,
                "attack_success_count": attack_success,
                "attack_success_rate_on_clean_correct": attack_success / clean_correct if clean_correct else 0.0,
                "recovered_count": stats["recovered"],
                "recovery_rate_on_successful_attacks": stats["recovered"] / attack_success if attack_success else 0.0,
                "still_failed_count": stats["still_failed"],
                "failure_rate_on_successful_attacks": stats["still_failed"] / attack_success if attack_success else 0.0,
            }
        )
    per_class_df = pd.DataFrame(rows)

    failure_rows = []
    for i, case in enumerate(failure_cases, start=1):
        failure_rows.append(
            {
                "case_id": i,
                "label": case["label"],
                "label_name": GTSRB_LABELS[case["label"]],
                "clean_pred": case["clean_pred"],
                "pgd_pred": case["adv_pred"],
                "jpeg_pred": case["def_pred"],
                "clean_conf": case["clean_conf"],
                "pgd_conf": case["adv_conf"],
                "jpeg_conf": case["def_conf"],
            }
        )
    failure_df = pd.DataFrame(failure_rows)

    plot_per_class(per_class_df, dirs["figures"])
    save_failure_grid(
        failure_cases,
        dirs["samples"] / "pgd_jpeg_failure_examples.png",
        "PGD epsilon=0.03 cases not recovered by JPEG Q75",
    )
    return per_class_df, failure_df


def copy_report_assets(dirs: dict[str, Path], config: dict) -> None:
    report_figure_dir = Path(config["outputs"].get("report_figure_dir", "reports/figures"))
    report_table_dir = Path(config["outputs"].get("report_table_dir", "reports/tables"))
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in {
        "per_class_attack_success_top.png": "fig_42_per_class_attack_success.png",
        "per_class_defense_recovery_top.png": "fig_43_per_class_defense_recovery.png",
    }.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)
    sample = dirs["samples"] / "pgd_jpeg_failure_examples.png"
    if sample.exists():
        shutil.copy2(sample, report_figure_dir / "fig_44_pgd_jpeg_failure_examples.png")
    for src_name, dst_name in {
        "per_class_robustness.csv": "table_16_per_class_robustness.csv",
        "defense_failure_cases.csv": "table_17_defense_failure_cases.csv",
    }.items():
        src = dirs["metrics"] / src_name
        if src.exists():
            shutil.copy2(src, report_table_dir / dst_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-class robustness and JPEG defense failures.")
    parser.add_argument("--config", type=Path, default=Path("configs/per_class_failure_analysis.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    result_dir = Path(config["outputs"]["result_dir"])
    dirs = ensure_output_dirs(result_dir)
    copy_config(args.config, dirs["root"])
    write_run_info(dirs, config, sys.argv)

    per_class_df, failure_df = evaluate(config, dirs)
    per_class_df.to_csv(dirs["metrics"] / "per_class_robustness.csv", index=False, encoding="utf-8-sig")
    failure_df.to_csv(dirs["metrics"] / "defense_failure_cases.csv", index=False, encoding="utf-8-sig")
    write_json(
        dirs["metrics"] / "per_class_failure_summary.json",
        {
            "top_vulnerable": per_class_df.sort_values("attack_success_rate_on_clean_correct", ascending=False)
            .head(8)
            .to_dict(orient="records"),
            "failure_cases": failure_df.to_dict(orient="records"),
        },
    )
    copy_report_assets(dirs, config)
    with (dirs["logs"] / "per_class_failure_analysis.log").open("w", encoding="utf-8") as f:
        f.write(per_class_df.to_string(index=False))
        f.write("\n\n")
        f.write(failure_df.to_string(index=False))
        f.write("\n")
    print(f"Per-class and failure analysis completed: {result_dir}")


if __name__ == "__main__":
    main()
