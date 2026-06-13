import argparse
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.attacks.methods import fgsm_attack, pgd_attack
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
    write_json,
    write_run_info,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_call(device: torch.device, fn) -> tuple[float, object]:
    sync(device)
    start = time.perf_counter()
    output = fn()
    sync(device)
    return time.perf_counter() - start, output


@torch.no_grad()
def run_inference(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images)


def benchmark(config: dict, model: torch.nn.Module, loader, device: torch.device) -> pd.DataFrame:
    bench_cfg = config["benchmark"]
    warmup_batches = int(bench_cfg.get("warmup_batches", 2))
    max_batches = int(bench_cfg.get("max_batches", 12))
    epsilon = float(bench_cfg.get("epsilon", 0.03))
    pgd_alpha = float(bench_cfg.get("pgd_alpha", 0.005))
    pgd_steps = int(bench_cfg.get("pgd_steps", 7))
    jpeg_quality = int(bench_cfg.get("jpeg_quality", 75))
    data_min, data_max = clamp_bounds(device)
    rows = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if batch_idx < warmup_batches:
            _ = run_inference(model, images)
            _ = fgsm_attack(model, images, labels, epsilon, data_min, data_max)
            _ = pgd_attack(model, images, labels, epsilon, pgd_alpha, pgd_steps, data_min, data_max)
            continue
        if batch_idx >= warmup_batches + max_batches:
            break

        batch_size = int(labels.numel())
        clean_seconds, _ = timed_call(device, lambda: run_inference(model, images))
        fgsm_seconds, fgsm_adv = timed_call(device, lambda: fgsm_attack(model, images, labels, epsilon, data_min, data_max))
        pgd_seconds, pgd_adv = timed_call(device, lambda: pgd_attack(model, images, labels, epsilon, pgd_alpha, pgd_steps, data_min, data_max))
        fgsm_rgb = denormalize(fgsm_adv)
        pgd_rgb = denormalize(pgd_adv)
        jpeg_fgsm_seconds, fgsm_jpeg = timed_call(device, lambda: jpeg_compress_batch(fgsm_rgb, jpeg_quality))
        jpeg_pgd_seconds, pgd_jpeg = timed_call(device, lambda: jpeg_compress_batch(pgd_rgb, jpeg_quality))
        jpeg_infer_seconds, _ = timed_call(device, lambda: run_inference(model, normalize(pgd_jpeg)))

        rows.extend(
            [
                {
                    "operation": "clean_inference",
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                    "seconds": clean_seconds,
                    "images_per_second": batch_size / clean_seconds,
                },
                {
                    "operation": "fgsm_generation",
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                    "seconds": fgsm_seconds,
                    "images_per_second": batch_size / fgsm_seconds,
                },
                {
                    "operation": "pgd_generation",
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                    "seconds": pgd_seconds,
                    "images_per_second": batch_size / pgd_seconds,
                },
                {
                    "operation": "jpeg_defense_fgsm",
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                    "seconds": jpeg_fgsm_seconds,
                    "images_per_second": batch_size / jpeg_fgsm_seconds,
                },
                {
                    "operation": "jpeg_defense_pgd",
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                    "seconds": jpeg_pgd_seconds,
                    "images_per_second": batch_size / jpeg_pgd_seconds,
                },
                {
                    "operation": "jpeg_pgd_plus_inference",
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                    "seconds": jpeg_pgd_seconds + jpeg_infer_seconds,
                    "images_per_second": batch_size / (jpeg_pgd_seconds + jpeg_infer_seconds),
                },
            ]
        )

    return pd.DataFrame(rows)


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for operation, group in raw_df.groupby("operation"):
        per_image_ms = group["seconds"] / group["batch_size"] * 1000.0
        rows.append(
            {
                "operation": operation,
                "batches": int(len(group)),
                "mean_batch_seconds": float(group["seconds"].mean()),
                "std_batch_seconds": float(group["seconds"].std(ddof=0)),
                "mean_ms_per_image": float(per_image_ms.mean()),
                "std_ms_per_image": float(per_image_ms.std(ddof=0)),
                "mean_images_per_second": float(group["images_per_second"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_ms_per_image")


def plot_summary(summary_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = summary_df.sort_values("mean_ms_per_image", ascending=True)
    ax.barh(ordered["operation"], ordered["mean_ms_per_image"])
    ax.set_title("Runtime Cost per Image")
    ax.set_xlabel("Milliseconds per image")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "runtime_ms_per_image.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["operation"], ordered["mean_images_per_second"])
    ax.set_title("Processing Throughput")
    ax.set_xlabel("Images per second")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "runtime_images_per_second.png", dpi=200)
    plt.close(fig)


def copy_report_assets(dirs: dict[str, Path], config: dict) -> None:
    report_figure_dir = Path(config["outputs"].get("report_figure_dir", "reports/figures"))
    report_table_dir = Path(config["outputs"].get("report_table_dir", "reports/tables"))
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in {
        "runtime_ms_per_image.png": "fig_40_runtime_ms_per_image.png",
        "runtime_images_per_second.png": "fig_41_runtime_images_per_second.png",
    }.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)
    table = dirs["metrics"] / "runtime_summary.csv"
    if table.exists():
        shutil.copy2(table, report_table_dir / "table_15_runtime_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark clean inference, attack generation, and JPEG defense runtime.")
    parser.add_argument("--config", type=Path, default=Path("configs/runtime_benchmark.yaml"))
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

    raw_df = benchmark(config, model, loader, device)
    summary_df = summarize(raw_df)
    raw_df.to_csv(dirs["metrics"] / "runtime_raw.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(dirs["metrics"] / "runtime_summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        dirs["metrics"] / "runtime_summary.json",
        {"metrics": summary_df.to_dict(orient="records")},
    )
    plot_summary(summary_df, dirs["figures"])
    copy_report_assets(dirs, config)
    with (dirs["logs"] / "runtime_benchmark.log").open("w", encoding="utf-8") as f:
        f.write(summary_df.to_string(index=False))
        f.write("\n")
    print(f"Runtime benchmark completed: {result_dir}")


if __name__ == "__main__":
    main()
