import argparse
import io
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
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


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images - mean) / std


def clamp_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = NORMALIZE_MEAN.to(device)
    std = NORMALIZE_STD.to(device)
    return (0.0 - mean) / std, (1.0 - mean) / std


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


def make_dataset(config: dict, dirs: dict[str, Path]) -> GTSRB | Subset:
    data_cfg = config["data"]
    dataset = GTSRB(
        root=data_cfg["root"],
        split="test",
        transform=build_eval_transform(int(data_cfg["image_size"])),
        download=False,
    )

    selected_path = data_cfg.get("selected_indices")
    if selected_path and Path(selected_path).exists():
        selected_indices = pd.read_csv(selected_path)["index"].astype(int).tolist()
        return Subset(dataset, selected_indices)

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
        return Subset(dataset, selected_indices)
    return dataset


def preprocess_pixels(images: torch.Tensor, method: str, params: dict) -> torch.Tensor:
    if method == "gaussian_blur":
        kernel_size = int(params.get("kernel_size", 3))
        sigma = float(params.get("sigma", 0.6))
        return TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    if method == "median_filter":
        kernel_size = int(params.get("kernel_size", 3))
        output = []
        for image in images.detach().cpu():
            arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            filtered = cv2.medianBlur(arr, kernel_size)
            output.append(torch.from_numpy(filtered).permute(2, 0, 1).float() / 255.0)
        return torch.stack(output, dim=0).to(images.device)

    if method == "jpeg_compression":
        quality = int(params.get("quality", 75))
        output = []
        for image in images.detach().cpu():
            arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            buffer = io.BytesIO()
            pil.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer).convert("RGB")
            output.append(torch.from_numpy(np.array(compressed)).permute(2, 0, 1).float() / 255.0)
        return torch.stack(output, dim=0).to(images.device)

    raise ValueError(f"Unsupported defense method: {method}")


@torch.no_grad()
def predict(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


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


def tensor_to_pil(tensor: torch.Tensor, size: int = 112) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def save_defense_grid(cases: list[dict], output_path: Path, title: str) -> None:
    if not cases:
        return
    cols = min(3, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    image_size = 112
    cell_w = 384
    cell_h = image_size + 90
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
        canvas.paste(tensor_to_pil(case["adversarial"], image_size), (x + 132, y))
        canvas.paste(tensor_to_pil(case["defended"], image_size), (x + 260, y))
        draw.text((x + 42, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 160, y + image_size + 5), "Attack", fill="red", font=font)
        draw.text((x + 286, y + image_size + 5), "Defense", fill="green", font=font)
        label = case["label"]
        draw.text((x + 4, y + image_size + 25), f"True: {label} {GTSRB_LABELS[label][:24]}", fill="black", font=font)
        draw.text((x + 4, y + image_size + 43), f"Adv pred: {case['adv_pred']} conf={case['adv_conf']:.3f}", fill="red", font=font)
        draw.text((x + 4, y + image_size + 61), f"Def pred: {case['def_pred']} conf={case['def_conf']:.3f}", fill="green", font=font)
    canvas.save(output_path)


def evaluate_defense(config: dict, model: torch.nn.Module, loader: DataLoader, device: torch.device, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    saved_cases: dict[str, list[dict]] = {}
    max_cases = int(config["outputs"].get("max_cases_per_grid", 9))

    for attack_name in ["fgsm", "pgd"]:
        for epsilon in config["attacks"][attack_name]["epsilons"]:
            epsilon = float(epsilon)
            totals = {
                method: {
                    "def_correct": 0,
                    "recovered": 0,
                    "def_conf": [],
                    "cases": [],
                }
                for method in config["defenses"]
            }
            total = 0
            clean_correct = 0
            adv_correct = 0
            attackable = 0

            for images, labels in tqdm(loader, desc=f"{attack_name} eps={epsilon}", leave=False, disable=True):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                clean_pred, _ = predict(model, images)
                adversarial = make_attack(model, images, labels, attack_name, epsilon, config, device)
                adv_pred, adv_conf = predict(model, adversarial)
                clean_rgb = denormalize(images)
                adv_rgb = denormalize(adversarial)

                total += int(labels.numel())
                clean_correct_mask = clean_pred == labels
                adv_correct_mask = adv_pred == labels
                clean_correct += int(clean_correct_mask.sum().item())
                adv_correct += int(adv_correct_mask.sum().item())
                attack_success_mask = clean_correct_mask & ~adv_correct_mask
                attackable += int(attack_success_mask.sum().item())

                for method, params in config["defenses"].items():
                    defended_rgb = preprocess_pixels(adv_rgb, method, params)
                    defended_norm = normalize(defended_rgb)
                    def_pred, def_conf = predict(model, defended_norm)
                    def_correct_mask = def_pred == labels
                    recovered_mask = attack_success_mask & def_correct_mask
                    totals[method]["def_correct"] += int(def_correct_mask.sum().item())
                    totals[method]["recovered"] += int(recovered_mask.sum().item())
                    totals[method]["def_conf"].extend(def_conf.detach().cpu().numpy().tolist())

                    if len(totals[method]["cases"]) < max_cases:
                        for idx in torch.where(recovered_mask)[0].detach().cpu().numpy().tolist():
                            if len(totals[method]["cases"]) >= max_cases:
                                break
                            totals[method]["cases"].append(
                                {
                                    "clean": clean_rgb[idx].detach().cpu(),
                                    "adversarial": adv_rgb[idx].detach().cpu(),
                                    "defended": defended_rgb[idx].detach().cpu(),
                                    "label": int(labels[idx].detach().cpu()),
                                    "adv_pred": int(adv_pred[idx].detach().cpu()),
                                    "def_pred": int(def_pred[idx].detach().cpu()),
                                    "adv_conf": float(adv_conf[idx].detach().cpu()),
                                    "def_conf": float(def_conf[idx].detach().cpu()),
                                }
                            )

            clean_accuracy = clean_correct / total
            adversarial_accuracy = adv_correct / total
            for method, values in totals.items():
                row = {
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "defense": method,
                    "clean_accuracy": clean_accuracy,
                    "adversarial_accuracy_before_defense": adversarial_accuracy,
                    "defended_accuracy": values["def_correct"] / total,
                    "recovery_rate_on_successful_attacks": values["recovered"] / attackable if attackable else 0.0,
                    "mean_defended_confidence": float(np.mean(values["def_conf"])),
                    "total_images": total,
                }
                rows.append(row)
                saved_cases[f"{attack_name}_{epsilon}_{method}"] = values["cases"]
                print(json.dumps(row, ensure_ascii=False))

                if epsilon == 0.03:
                    save_defense_grid(
                        values["cases"],
                        dirs["samples"] / f"{attack_name}_eps_0.03_{method}_recovered.png",
                        f"{method} recovered examples, {attack_name.upper()} eps=0.03",
                    )

    return pd.DataFrame(rows)


def plot_defense_results(metrics_df: pd.DataFrame, figure_dir: Path) -> None:
    for attack in metrics_df["attack"].unique():
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = metrics_df[metrics_df["attack"] == attack]
        for defense, group in subset.groupby("defense"):
            ax.plot(group["epsilon"], group["defended_accuracy"], marker="o", label=defense)
        before = subset.groupby("epsilon")["adversarial_accuracy_before_defense"].first().reset_index()
        ax.plot(before["epsilon"], before["adversarial_accuracy_before_defense"], marker="x", linestyle="--", label="before_defense")
        ax.set_title(f"Input Preprocessing Defense under {attack.upper()}")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{attack}_defense_accuracy_curve.png", dpi=200)
        plt.close(fig)

    eps_subset = metrics_df[np.isclose(metrics_df["epsilon"], 0.03)]
    labels = []
    values = []
    for _, row in eps_subset.iterrows():
        labels.append(f"{row['attack']}:{row['defense']}")
        values.append(row["defended_accuracy"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    before_rows = eps_subset.groupby("attack")["adversarial_accuracy_before_defense"].first()
    for attack, value in before_rows.items():
        ax.axhline(value, linestyle="--", linewidth=1, alpha=0.6, label=f"{attack} before={value:.3f}")
    ax.set_title("Input Preprocessing Defense Accuracy at Epsilon=0.03")
    ax.set_xlabel("Attack and defense")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "input_defense_accuracy_bar_eps003.png", dpi=200)
    plt.close(fig)


def copy_report_assets(dirs: dict[str, Path], report_figure_dir: Path, report_table_dir: Path) -> None:
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "fgsm_defense_accuracy_curve.png": "fig_16_fgsm_input_defense_curve.png",
        "pgd_defense_accuracy_curve.png": "fig_17_pgd_input_defense_curve.png",
        "input_defense_accuracy_bar_eps003.png": "fig_18_input_defense_accuracy_bar.png",
    }
    for src_name, dst_name in figure_map.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / dst_name)

    sample_candidates = [
        "fgsm_eps_0.03_gaussian_blur_recovered.png",
        "pgd_eps_0.03_gaussian_blur_recovered.png",
        "fgsm_eps_0.03_median_filter_recovered.png",
    ]
    for offset, src_name in enumerate(sample_candidates, start=19):
        src = dirs["samples"] / src_name
        if src.exists():
            shutil.copy2(src, report_figure_dir / f"fig_{offset}_input_defense_examples.png")
            break

    metrics = dirs["metrics"] / "input_defense_metrics.csv"
    if metrics.exists():
        shutil.copy2(metrics, report_table_dir / "table_07_input_defense_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate input preprocessing defenses.")
    parser.add_argument("--config", type=Path, default=Path("configs/defense_input_preprocessing.yaml"))
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

    dataset = make_dataset(config, dirs)
    loader_kwargs = {
        "batch_size": int(config["data"]["batch_size"]),
        "shuffle": False,
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    model = load_model(Path(config["model"]["checkpoint"]), device)

    metrics_df = evaluate_defense(config, model, loader, device, dirs)
    metrics_df.to_csv(dirs["metrics"] / "input_defense_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["metrics"] / "input_defense_summary.json", {"metrics": metrics_df.to_dict(orient="records")})
    plot_defense_results(metrics_df, dirs["figures"])
    copy_report_assets(
        dirs,
        Path(config["outputs"].get("report_figure_dir", "reports/figures")),
        Path(config["outputs"].get("report_table_dir", "reports/tables")),
    )
    with (dirs["logs"] / "input_defense.log").open("w", encoding="utf-8") as f:
        f.write(metrics_df.to_string(index=False))
        f.write("\n")
    print(f"Input preprocessing defense completed: {result_dir}")


if __name__ == "__main__":
    main()

