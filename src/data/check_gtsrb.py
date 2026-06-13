import argparse
import json
import math
import platform
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import GTSRB

from src.data.gtsrb_labels import GTSRB_LABELS


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(result_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": result_dir,
        "metrics": result_dir / "metrics",
        "figures": result_dir / "figures",
        "samples": result_dir / "samples",
        "logs": result_dir / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_datasets(root: Path, download: bool) -> tuple[GTSRB, GTSRB]:
    train_dataset = GTSRB(root=str(root), split="train", download=download)
    test_dataset = GTSRB(root=str(root), split="test", download=download)
    return train_dataset, test_dataset


def collect_records(dataset: GTSRB, split: str) -> list[dict]:
    records = []
    for path, label in dataset._samples:
        with Image.open(path) as img:
            width, height = img.size
        records.append(
            {
                "split": split,
                "path": path,
                "class_id": int(label),
                "class_name": GTSRB_LABELS[int(label)],
                "width": width,
                "height": height,
                "area": width * height,
            }
        )
    return records


def plot_class_distribution(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(summary))
    width = 0.42
    ax.bar(x - width / 2, summary["train_count"], width, label="Train")
    ax.bar(x + width / 2, summary["test_count"], width, label="Test")
    ax.set_title("GTSRB Class Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Number of images")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["class_id"], rotation=90)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_image_size_distribution(records: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(records["width"], records["height"], s=9, alpha=0.35)
    axes[0].set_title("Original Image Width vs Height")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].grid(alpha=0.25)

    axes[1].hist(records["area"], bins=40)
    axes[1].set_title("Image Area Distribution")
    axes[1].set_xlabel("Width x Height")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def select_balanced_samples(dataset: GTSRB, count: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[str]] = {}
    for path, label in dataset._samples:
        by_class.setdefault(int(label), []).append(path)

    selected = []
    class_ids = list(sorted(by_class))
    rng.shuffle(class_ids)
    for class_id in class_ids:
        if len(selected) >= count:
            break
        selected.append((rng.choice(by_class[class_id]), class_id))

    while len(selected) < count:
        path, label = rng.choice(dataset._samples)
        selected.append((path, int(label)))

    return selected[:count]


def make_sample_grid(
    samples: list[tuple[str, int]],
    output_path: Path,
    title: str,
    image_size: int,
) -> None:
    cols = min(6, len(samples))
    rows = math.ceil(len(samples) / cols)
    label_h = 48
    title_h = 52
    cell_w = max(178, image_size + 72)
    cell_h = image_size + label_h + 14
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 16), title, fill="black", font=font)

    for idx, (path, label) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = title_h + row * cell_h
        image_x = x + (cell_w - image_size) // 2
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((image_size - 8, image_size - 8), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (image_size, image_size), "white")
            tile.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
        canvas.paste(tile, (image_x, y))
        draw.rectangle((image_x, y, image_x + image_size - 1, y + image_size - 1), outline=(220, 220, 220))

        label_line_1 = f"Class {label}"
        label_line_2 = GTSRB_LABELS[label]
        if len(label_line_2) > 24:
            label_line_2 = label_line_2[:23] + "."
        text_y = y + image_size + 6
        draw.text((x + 8, text_y), label_line_1, fill="black", font=font)
        draw.text((x + 8, text_y + 18), label_line_2, fill="black", font=font)

    canvas.save(output_path)


def save_summary_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    train_counts = Counter(train_df["class_id"])
    test_counts = Counter(test_df["class_id"])
    rows = []
    for class_id in range(43):
        rows.append(
            {
                "class_id": class_id,
                "class_name": GTSRB_LABELS[class_id],
                "train_count": train_counts.get(class_id, 0),
                "test_count": test_counts.get(class_id, 0),
                "total_count": train_counts.get(class_id, 0) + test_counts.get(class_id, 0),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    full_records = pd.concat([train_df, test_df], ignore_index=True)
    full_records.to_csv(output_dir / "image_records.csv", index=False, encoding="utf-8-sig")

    overview = {
        "train_images": int(len(train_df)),
        "test_images": int(len(test_df)),
        "total_images": int(len(full_records)),
        "num_classes": 43,
        "min_width": int(full_records["width"].min()),
        "max_width": int(full_records["width"].max()),
        "min_height": int(full_records["height"].min()),
        "max_height": int(full_records["height"].max()),
        "mean_width": float(full_records["width"].mean()),
        "mean_height": float(full_records["height"].mean()),
    }
    write_json(output_dir / "dataset_overview.json", overview)
    return summary


def copy_report_assets(paths: dict[str, Path], report_figure_dir: Path, report_table_dir: Path) -> None:
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)

    figure_map = {
        "class_distribution": "fig_01_class_distribution.png",
        "image_size_distribution": "fig_02_image_size_distribution.png",
        "train_grid": "fig_03_train_samples.png",
        "test_grid": "fig_04_test_samples.png",
    }
    for key, name in figure_map.items():
        shutil.copy2(paths[key], report_figure_dir / name)

    table_map = {
        "dataset_summary": "table_01_dataset_summary.csv",
        "dataset_overview": "table_02_dataset_overview.json",
    }
    for key, name in table_map.items():
        shutil.copy2(paths[key], report_table_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and visualize the GTSRB dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_check.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    result_dir = args.output_dir or Path(config["outputs"]["result_dir"])
    dirs = ensure_dirs(result_dir)

    shutil.copy2(args.config, dirs["root"] / "config.yaml")
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
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
        },
    )

    data_cfg = config["data"]
    download = bool(data_cfg.get("download", True)) and not args.no_download
    train_dataset, test_dataset = load_datasets(Path(data_cfg["root"]), download)

    train_df = pd.DataFrame(collect_records(train_dataset, "train"))
    test_df = pd.DataFrame(collect_records(test_dataset, "test"))
    summary = save_summary_tables(train_df, test_df, dirs["metrics"])

    class_dist_path = dirs["figures"] / "class_distribution.png"
    size_dist_path = dirs["figures"] / "image_size_distribution.png"
    train_grid_path = dirs["samples"] / "sample_grid_train.png"
    test_grid_path = dirs["samples"] / "sample_grid_test.png"

    plot_class_distribution(summary, class_dist_path)
    plot_image_size_distribution(pd.concat([train_df, test_df], ignore_index=True), size_dist_path)

    sample_count = int(data_cfg.get("sample_grid_per_split", 36))
    image_size = int(data_cfg.get("image_size_preview", 96))
    make_sample_grid(
        select_balanced_samples(train_dataset, sample_count, seed),
        train_grid_path,
        "GTSRB Train Samples",
        image_size,
    )
    make_sample_grid(
        select_balanced_samples(test_dataset, sample_count, seed + 1),
        test_grid_path,
        "GTSRB Test Samples",
        image_size,
    )

    paths = {
        "class_distribution": class_dist_path,
        "image_size_distribution": size_dist_path,
        "train_grid": train_grid_path,
        "test_grid": test_grid_path,
        "dataset_summary": dirs["metrics"] / "dataset_summary.csv",
        "dataset_overview": dirs["metrics"] / "dataset_overview.json",
    }
    copy_report_assets(
        paths,
        Path(config["outputs"]["report_figure_dir"]),
        Path(config["outputs"]["report_table_dir"]),
    )

    log_path = dirs["logs"] / "dataset_check.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Train images: {len(train_dataset)}\n")
        f.write(f"Test images: {len(test_dataset)}\n")
        f.write(f"Class count: {summary.shape[0]}\n")
        f.write("Generated dataset summary, distribution plots, and sample grids.\n")

    print(f"Dataset check completed: {result_dir}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Report figures copied to: {config['outputs']['report_figure_dir']}")


if __name__ == "__main__":
    main()
