import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from src.models.classifiers import build_model

NORMALIZE_MEAN = torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1)
NORMALIZE_STD = torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_output_dirs(result_dir: Path, include_checkpoints: bool = False) -> dict[str, Path]:
    dirs = {
        "root": result_dir,
        "logs": result_dir / "logs",
        "metrics": result_dir / "metrics",
        "figures": result_dir / "figures",
        "samples": result_dir / "samples",
    }
    if include_checkpoints:
        dirs["checkpoints"] = result_dir / "checkpoints"
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_config(config_path: Path, output_root: Path) -> None:
    shutil.copy2(config_path, output_root / "config.yaml")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_run_info(dirs: dict[str, Path], config: dict, command: list[str]) -> None:
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
            "command": " ".join(command),
        },
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images - mean) / std


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


def clamp_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = NORMALIZE_MEAN.to(device)
    std = NORMALIZE_STD.to(device)
    return (0.0 - mean) / std, (1.0 - mean) / std


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Run `python -m src.train_classifier --config configs/baseline_resnet18.yaml` first, "
            "or update the config to point to an existing checkpoint."
        )
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


def make_test_dataset(config: dict, dirs: dict[str, Path] | None = None) -> GTSRB | Subset:
    data_cfg = config["data"]
    dataset = GTSRB(
        root=data_cfg.get("root", "data/raw"),
        split="test",
        transform=build_eval_transform(int(data_cfg["image_size"])),
        download=bool(data_cfg.get("download", False)),
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
        if dirs is not None:
            pd.DataFrame({"index": selected_indices}).to_csv(
                dirs["metrics"] / "selected_eval_indices.csv",
                index=False,
                encoding="utf-8-sig",
            )
        return Subset(dataset, selected_indices)
    return dataset


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


@torch.no_grad()
def predict(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf, probs


def tensor_to_pil(tensor: torch.Tensor, size: int | None = None) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(arr)
    if size is not None:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def noise_map(clean_rgb: torch.Tensor, changed_rgb: torch.Tensor) -> torch.Tensor:
    noise = (changed_rgb - clean_rgb).abs().mean(dim=0)
    if float(noise.max()) > 0:
        noise = noise / noise.max()
    return noise
