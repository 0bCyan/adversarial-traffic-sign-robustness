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

from src.attacks.methods import fgsm_attack, pgd_attack
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


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        class_id = int(pred.item()) if target_class is None else int(target_class)
        score = logits[:, class_id].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, int(pred.item()), float(conf.item())


def get_target_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in layer_name.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def make_attack(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, config: dict, device: torch.device) -> torch.Tensor:
    attack_cfg = config["attack"]
    method = attack_cfg["method"].lower()
    epsilon = float(attack_cfg["epsilon"])
    data_min, data_max = clamp_bounds(device)
    if method == "fgsm":
        return fgsm_attack(model, images, labels, epsilon, data_min, data_max)
    if method == "pgd":
        return pgd_attack(
            model,
            images,
            labels,
            epsilon,
            float(attack_cfg.get("alpha", 0.005)),
            int(attack_cfg.get("steps", 7)),
            data_min,
            data_max,
        )
    raise ValueError(f"Unsupported attack method: {method}")


def overlay_cam(image_rgb: torch.Tensor, cam: np.ndarray, size: int = 144) -> Image.Image:
    image = tensor_to_pil(image_rgb, size=size).convert("RGB")
    heat = (plt.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize((size, size), Image.Resampling.BILINEAR).convert("RGB")
    return Image.blend(image, heat_img, alpha=0.42)


def collect_cases(
    config: dict,
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> list[dict]:
    max_cases = int(config["gradcam"].get("max_cases", 6))
    prefer_recovered = bool(config["gradcam"].get("prefer_recovered_cases", True))
    cases = []
    fallback_cases = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_pred, clean_conf, _ = predict(model, images)
        adversarial = make_attack(model, images, labels, config, device)
        adv_pred, adv_conf, _ = predict(model, adversarial)

        adv_rgb = denormalize(adversarial)
        defended_rgb = preprocess_pixels(adv_rgb, config["defense"]["method"], config["defense"])
        defended_norm = normalize(defended_rgb)
        def_pred, def_conf, _ = predict(model, defended_norm)
        clean_rgb = denormalize(images)

        clean_correct = clean_pred == labels
        attack_success = clean_correct & (adv_pred != labels)
        recovered = attack_success & (def_pred == labels)
        candidate_mask = recovered if prefer_recovered else attack_success
        fallback_mask = attack_success

        for idx in torch.where(candidate_mask)[0].detach().cpu().numpy().tolist():
            if len(cases) >= max_cases:
                break
            cases.append(
                {
                    "clean": clean_rgb[idx].detach().cpu(),
                    "adversarial": adv_rgb[idx].detach().cpu(),
                    "defended": defended_rgb[idx].detach().cpu(),
                    "clean_norm": images[idx : idx + 1].detach(),
                    "adv_norm": adversarial[idx : idx + 1].detach(),
                    "def_norm": defended_norm[idx : idx + 1].detach(),
                    "label": int(labels[idx].detach().cpu()),
                    "clean_pred": int(clean_pred[idx].detach().cpu()),
                    "adv_pred": int(adv_pred[idx].detach().cpu()),
                    "def_pred": int(def_pred[idx].detach().cpu()),
                    "clean_conf": float(clean_conf[idx].detach().cpu()),
                    "adv_conf": float(adv_conf[idx].detach().cpu()),
                    "def_conf": float(def_conf[idx].detach().cpu()),
                    "recovered": bool(recovered[idx].detach().cpu()),
                }
            )

        for idx in torch.where(fallback_mask)[0].detach().cpu().numpy().tolist():
            if len(fallback_cases) >= max_cases:
                break
            fallback_cases.append(
                {
                    "clean": clean_rgb[idx].detach().cpu(),
                    "adversarial": adv_rgb[idx].detach().cpu(),
                    "defended": defended_rgb[idx].detach().cpu(),
                    "clean_norm": images[idx : idx + 1].detach(),
                    "adv_norm": adversarial[idx : idx + 1].detach(),
                    "def_norm": defended_norm[idx : idx + 1].detach(),
                    "label": int(labels[idx].detach().cpu()),
                    "clean_pred": int(clean_pred[idx].detach().cpu()),
                    "adv_pred": int(adv_pred[idx].detach().cpu()),
                    "def_pred": int(def_pred[idx].detach().cpu()),
                    "clean_conf": float(clean_conf[idx].detach().cpu()),
                    "adv_conf": float(adv_conf[idx].detach().cpu()),
                    "def_conf": float(def_conf[idx].detach().cpu()),
                    "recovered": bool(recovered[idx].detach().cpu()),
                }
            )
        if len(cases) >= max_cases:
            break

    if not cases:
        cases = fallback_cases[:max_cases]
    return cases[:max_cases]


def save_gradcam_grid(cases: list[dict], output_path: Path, title: str, gradcam: GradCAM, device: torch.device) -> pd.DataFrame:
    if not cases:
        return pd.DataFrame()

    image_size = 144
    cell_w = 204
    cell_h = image_size + 86
    title_h = 48
    cols = 3
    rows = len(cases)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 15), title, fill="black", font=font)
    for col, text in enumerate(["Clean Grad-CAM", "Attack Grad-CAM", "Defense Grad-CAM"]):
        draw.text((col * cell_w + 42, title_h - 18), text, fill="black", font=font)

    rows_out = []
    for row, case in enumerate(cases):
        y = title_h + row * cell_h
        clean_cam, _, _ = gradcam(case["clean_norm"].to(device), case["clean_pred"])
        adv_cam, _, _ = gradcam(case["adv_norm"].to(device), case["adv_pred"])
        def_cam, _, _ = gradcam(case["def_norm"].to(device), case["def_pred"])

        images = [
            overlay_cam(case["clean"], clean_cam, image_size),
            overlay_cam(case["adversarial"], adv_cam, image_size),
            overlay_cam(case["defended"], def_cam, image_size),
        ]
        for col, image in enumerate(images):
            x = col * cell_w + 30
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + image_size - 1, y + image_size - 1), outline=(215, 215, 215))

        label = case["label"]
        meta = [
            f"T {label}: {GTSRB_LABELS[label][:20]}",
            f"C {case['clean_pred']} {case['clean_conf']:.3f}",
            f"A {case['adv_pred']} {case['adv_conf']:.3f}",
            f"D {case['def_pred']} {case['def_conf']:.3f}",
        ]
        text_y = y + image_size + 8
        draw.text((8, text_y), meta[0], fill="black", font=font)
        draw.text((8, text_y + 18), meta[1], fill="green", font=font)
        draw.text((cell_w + 8, text_y + 18), meta[2], fill="red", font=font)
        draw.text((2 * cell_w + 8, text_y + 18), meta[3], fill="green" if case["recovered"] else "black", font=font)

        rows_out.append(
            {
                "case_id": row + 1,
                "label": label,
                "label_name": GTSRB_LABELS[label],
                "clean_pred": case["clean_pred"],
                "adv_pred": case["adv_pred"],
                "def_pred": case["def_pred"],
                "clean_conf": case["clean_conf"],
                "adv_conf": case["adv_conf"],
                "def_conf": case["def_conf"],
                "recovered_by_defense": case["recovered"],
            }
        )

    canvas.save(output_path)
    return pd.DataFrame(rows_out)


def copy_report_assets(dirs: dict[str, Path], config: dict) -> None:
    report_figure_dir = Path(config["outputs"].get("report_figure_dir", "reports/figures"))
    report_table_dir = Path(config["outputs"].get("report_table_dir", "reports/tables"))
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)
    figure = dirs["figures"] / "gradcam_clean_attack_defense.png"
    if figure.exists():
        shutil.copy2(figure, report_figure_dir / "fig_33_gradcam_clean_attack_defense.png")
    table = dirs["metrics"] / "gradcam_cases.csv"
    if table.exists():
        shutil.copy2(table, report_table_dir / "table_13_gradcam_cases.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM clean/attack/defense case studies.")
    parser.add_argument("--config", type=Path, default=Path("configs/explainability_gradcam.yaml"))
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

    cases = collect_cases(config, model, loader, device)
    target_layer = get_target_layer(model, config["gradcam"].get("target_layer", "layer4"))
    gradcam = GradCAM(model, target_layer)
    try:
        case_df = save_gradcam_grid(
            cases,
            dirs["figures"] / "gradcam_clean_attack_defense.png",
            "Grad-CAM: clean vs adversarial vs JPEG defense",
            gradcam,
            device,
        )
    finally:
        gradcam.close()

    case_df.to_csv(dirs["metrics"] / "gradcam_cases.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["metrics"] / "gradcam_summary.json", {"cases": case_df.to_dict(orient="records")})
    copy_report_assets(dirs, config)
    with (dirs["logs"] / "gradcam.log").open("w", encoding="utf-8") as f:
        f.write(case_df.to_string(index=False))
        f.write("\n")
    print(f"Grad-CAM analysis completed: {result_dir}")


if __name__ == "__main__":
    main()
