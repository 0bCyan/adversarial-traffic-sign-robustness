import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from src.attacks.methods import fgsm_attack, pgd_attack
from src.data.gtsrb_labels import GTSRB_LABELS
from src.evaluate_attacks import load_model

NORMALIZE_MEAN = torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1)
NORMALIZE_STD = torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1)


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


def clamp_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = NORMALIZE_MEAN.to(device)
    std = NORMALIZE_STD.to(device)
    return (0.0 - mean) / std, (1.0 - mean) / std


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.to(images.device)
    std = NORMALIZE_STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


@torch.no_grad()
def predict(model: torch.nn.Module, images: torch.Tensor):
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def make_attack(model, images, labels, attack, epsilon, device, pgd_alpha, pgd_steps):
    data_min, data_max = clamp_bounds(device)
    if attack == "fgsm":
        return fgsm_attack(model, images, labels, epsilon, data_min, data_max)
    if attack == "pgd":
        return pgd_attack(model, images, labels, epsilon, pgd_alpha, pgd_steps, data_min, data_max)
    raise ValueError(attack)


def tensor_to_pil(tensor: torch.Tensor, size: int = 128) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def amplified_delta_to_pil(clean: torch.Tensor, adv: torch.Tensor, scale: float = 30.0, size: int = 128) -> Image.Image:
    delta = (adv - clean).detach().cpu()
    # Center at neutral gray so positive/negative differences are visible.
    visible = (0.5 + delta * scale).clamp(0, 1)
    arr = (visible.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def heatmap_delta_to_pil(clean: torch.Tensor, adv: torch.Tensor, size: int = 128) -> Image.Image:
    delta = (adv - clean).detach().cpu().abs().mean(dim=0)
    if float(delta.max()) > 0:
        delta = delta / delta.max()
    arr = (plt.get_cmap("magma")(delta.numpy())[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)


def save_perceptual_grid(cases: list[dict], out_path: Path, title: str):
    if not cases:
        return
    cols = min(2, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    image_size = 128
    cell_w = 590
    cell_h = image_size + 100
    title_h = 42
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 14), title, fill="black", font=font)
    for idx, case in enumerate(cases):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = title_h + row * cell_h
        clean = case["clean"]
        adv = case["adv"]
        canvas.paste(tensor_to_pil(clean, image_size), (x + 6, y))
        canvas.paste(tensor_to_pil(adv, image_size), (x + 152, y))
        canvas.paste(amplified_delta_to_pil(clean, adv, scale=30, size=image_size), (x + 298, y))
        canvas.paste(heatmap_delta_to_pil(clean, adv, image_size), (x + 444, y))
        draw.text((x + 46, y + image_size + 5), "Original", fill="black", font=font)
        draw.text((x + 184, y + image_size + 5), "Adversarial", fill="black", font=font)
        draw.text((x + 320, y + image_size + 5), "Delta x30", fill="black", font=font)
        draw.text((x + 474, y + image_size + 5), "Delta heat", fill="black", font=font)
        label = case["label"]
        draw.text((x + 6, y + image_size + 26), f"True {label}: {GTSRB_LABELS[label][:30]}", fill="black", font=font)
        draw.text((x + 6, y + image_size + 44), f"Clean pred {case['clean_pred']} conf={case['clean_conf']:.3f}", fill="green", font=font)
        draw.text((x + 6, y + image_size + 62), f"Adv pred {case['adv_pred']} conf={case['adv_conf']:.3f}", fill="red", font=font)
        draw.text((x + 6, y + image_size + 80), f"PSNR={case['psnr']:.2f} dB, Linf={case['linf']:.4f}", fill="black", font=font)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/01_baseline/resnet18/checkpoints/best_model.pth"))
    parser.add_argument("--indices", type=Path, default=Path("results/02_attack/fgsm_pgd/metrics/selected_eval_indices.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/02_attack/perturbation_analysis"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_metrics = args.out_dir / "metrics"
    out_figures = args.out_dir / "figures"
    out_samples = args.out_dir / "samples"
    for path in (out_metrics, out_figures, out_samples):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GTSRB(root="data/raw", split="test", transform=build_transform(args.image_size), download=False)
    if args.indices.exists():
        indices = pd.read_csv(args.indices)["index"].astype(int).tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    model = load_model(args.checkpoint, device)
    configs = [
        ("fgsm", 0.01),
        ("fgsm", 0.03),
        ("pgd", 0.01),
        ("pgd", 0.03),
    ]
    rows = []
    cases_by_key: dict[str, list[dict]] = {f"{a}_{e}": [] for a, e in configs}
    for attack, epsilon in configs:
        total = 0
        l_inf_values = []
        l1_values = []
        mse_values = []
        psnr_values = []
        changed = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_pred, clean_conf = predict(model, images)
            adv_norm = make_attack(model, images, labels, attack, epsilon, device, pgd_alpha=0.005, pgd_steps=7)
            adv_pred, adv_conf = predict(model, adv_norm)
            clean = denormalize(images)
            adv = denormalize(adv_norm)
            delta = adv - clean
            batch_linf = delta.abs().flatten(1).max(dim=1).values
            batch_l1 = delta.abs().flatten(1).mean(dim=1)
            batch_mse = (delta.pow(2)).flatten(1).mean(dim=1)
            batch_psnr = 10 * torch.log10(1.0 / torch.clamp(batch_mse, min=1e-12))
            l_inf_values.extend(batch_linf.detach().cpu().numpy().tolist())
            l1_values.extend(batch_l1.detach().cpu().numpy().tolist())
            mse_values.extend(batch_mse.detach().cpu().numpy().tolist())
            psnr_values.extend(batch_psnr.detach().cpu().numpy().tolist())
            total += int(labels.numel())
            changed_mask = (clean_pred == labels) & (adv_pred != labels)
            changed += int(changed_mask.sum().item())
            key = f"{attack}_{epsilon}"
            if len(cases_by_key[key]) < 6:
                for idx in torch.where(changed_mask)[0].detach().cpu().numpy().tolist():
                    if len(cases_by_key[key]) >= 6:
                        break
                    cases_by_key[key].append(
                        {
                            "clean": clean[idx].detach().cpu(),
                            "adv": adv[idx].detach().cpu(),
                            "label": int(labels[idx].detach().cpu()),
                            "clean_pred": int(clean_pred[idx].detach().cpu()),
                            "adv_pred": int(adv_pred[idx].detach().cpu()),
                            "clean_conf": float(clean_conf[idx].detach().cpu()),
                            "adv_conf": float(adv_conf[idx].detach().cpu()),
                            "psnr": float(batch_psnr[idx].detach().cpu()),
                            "linf": float(batch_linf[idx].detach().cpu()),
                        }
                    )
        rows.append(
            {
                "attack": attack,
                "epsilon_normalized_space": epsilon,
                "mean_linf_pixel_space": float(np.mean(l_inf_values)),
                "max_linf_pixel_space": float(np.max(l_inf_values)),
                "mean_abs_delta_pixel_space": float(np.mean(l1_values)),
                "mean_mse": float(np.mean(mse_values)),
                "mean_psnr_db": float(np.mean(psnr_values)),
                "min_psnr_db": float(np.min(psnr_values)),
                "successful_changes": changed,
                "total_images": total,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_metrics / "perturbation_perceptibility_metrics.csv", index=False, encoding="utf-8-sig")

    for attack, epsilon in configs:
        key = f"{attack}_{epsilon}"
        if epsilon == 0.03:
            save_perceptual_grid(
                cases_by_key[key],
                out_samples / f"{attack}_eps_0.03_perceptual_grid.png",
                f"{attack.upper()} epsilon={epsilon}: original vs adversarial vs amplified perturbation",
            )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{r['attack'].upper()}\neps={r['epsilon_normalized_space']}" for r in rows]
    ax.bar(labels, [r["mean_psnr_db"] for r in rows])
    ax.set_title("Perturbation Perceptibility: Mean PSNR")
    ax.set_ylabel("PSNR (dB), higher means less visible")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_figures / "perturbation_psnr_bar.png", dpi=200)
    plt.close(fig)

    # Copy report assets.
    report_fig = Path("reports/figures")
    report_tbl = Path("reports/tables")
    report_fig.mkdir(parents=True, exist_ok=True)
    report_tbl.mkdir(parents=True, exist_ok=True)
    (out_metrics / "perturbation_perceptibility_metrics.csv").replace(report_tbl / "table_08_perturbation_perceptibility_metrics.csv")
    # Keep a copy in results after replace.
    df.to_csv(out_metrics / "perturbation_perceptibility_metrics.csv", index=False, encoding="utf-8-sig")
    for src, dst in [
        (out_figures / "perturbation_psnr_bar.png", report_fig / "fig_20_perturbation_psnr_bar.png"),
        (out_samples / "fgsm_eps_0.03_perceptual_grid.png", report_fig / "fig_21_fgsm_perceptual_grid.png"),
        (out_samples / "pgd_eps_0.03_perceptual_grid.png", report_fig / "fig_22_pgd_perceptual_grid.png"),
    ]:
        if src.exists():
            dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
