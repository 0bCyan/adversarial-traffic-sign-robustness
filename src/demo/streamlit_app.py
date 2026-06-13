import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.attacks.methods import fgsm_attack, pgd_attack
from src.data.gtsrb_labels import GTSRB_LABELS
from src.defenses.preprocessing import jpeg_compress_batch
from src.experiment_utils import clamp_bounds, denormalize, load_checkpoint_model, normalize, noise_map, predict

CONFIG_PATH = Path("configs/demo_streamlit.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "data": {"image_size": 64},
        "model": {"checkpoint": "results/01_baseline/resnet18/checkpoints/best_model.pth"},
        "attack": {"default_method": "fgsm", "default_epsilon": 0.03, "pgd_alpha": 0.005, "pgd_steps": 7},
        "defense": {"jpeg_quality": 75},
    }


APP_CONFIG = load_config()
CHECKPOINT_PATH = Path(APP_CONFIG["model"]["checkpoint"])
IMAGE_SIZE = int(APP_CONFIG["data"].get("image_size", 64))


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)),
        ]
    )


@st.cache_resource
def load_model() -> torch.nn.Module:
    return load_checkpoint_model(CHECKPOINT_PATH, DEVICE)


def to_display_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clamp(0, 1).squeeze(0)
    arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((192, 192), Image.Resampling.LANCZOS)


def to_heatmap_image(clean_rgb: torch.Tensor, adv_rgb: torch.Tensor) -> Image.Image:
    heat = noise_map(clean_rgb.squeeze(0), adv_rgb.squeeze(0)).numpy()
    arr = (plt.get_cmap("magma")(heat)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((192, 192), Image.Resampling.LANCZOS)


def prediction_table(probs: torch.Tensor) -> list[dict]:
    values, indices = probs.squeeze(0).detach().cpu().topk(5)
    rows = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "class_id": int(index),
                "class_name": GTSRB_LABELS[int(index)],
                "confidence": float(value),
            }
        )
    return rows


def make_attack(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    method: str,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
) -> torch.Tensor:
    data_min, data_max = clamp_bounds(DEVICE)
    if method == "FGSM":
        return fgsm_attack(model, image, label, epsilon, data_min, data_max)
    return pgd_attack(model, image, label, epsilon, pgd_alpha, pgd_steps, data_min, data_max)


def render_prediction(name: str, pred: torch.Tensor, conf: torch.Tensor, probs: torch.Tensor) -> None:
    class_id = int(pred.item())
    st.metric(name, f"{class_id} - {GTSRB_LABELS[class_id]}", f"{float(conf.item()):.3f}")
    st.dataframe(prediction_table(probs), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Traffic Sign Robustness Demo", layout="wide")
    st.title("Traffic Sign Robustness Demo")

    if not CHECKPOINT_PATH.exists():
        st.error(f"Missing model checkpoint: {CHECKPOINT_PATH}")
        st.code("python -m src.train_classifier --config configs/baseline_resnet18.yaml", language="bash")
        st.stop()

    model = load_model()
    transform = build_transform(IMAGE_SIZE)
    attack_cfg = APP_CONFIG.get("attack", {})
    defense_cfg = APP_CONFIG.get("defense", {})
    default_attack = str(attack_cfg.get("default_method", "fgsm")).upper()
    attack_options = ["FGSM", "PGD"]

    with st.sidebar:
        uploaded = st.file_uploader("Upload traffic sign image", type=["png", "jpg", "jpeg", "bmp"])
        attack_method = st.selectbox("Attack", attack_options, index=attack_options.index(default_attack) if default_attack in attack_options else 0)
        epsilon = st.slider(
            "Epsilon",
            min_value=0.0,
            max_value=0.08,
            value=float(attack_cfg.get("default_epsilon", 0.03)),
            step=0.005,
            format="%.3f",
        )
        pgd_steps = st.slider("PGD steps", min_value=1, max_value=20, value=int(attack_cfg.get("pgd_steps", 7)), step=1)
        pgd_alpha = st.slider(
            "PGD alpha",
            min_value=0.001,
            max_value=0.02,
            value=float(attack_cfg.get("pgd_alpha", 0.005)),
            step=0.001,
            format="%.3f",
        )
        use_jpeg = st.checkbox("JPEG defense", value=True)
        jpeg_quality = st.slider("JPEG quality", min_value=30, max_value=100, value=int(defense_cfg.get("jpeg_quality", 75)), step=5)

    if uploaded is None:
        st.info("Upload a traffic sign image to run the attack and defense pipeline.")
        st.stop()

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    clean_pred, clean_conf, clean_probs = predict(model, input_tensor)
    label_for_attack = clean_pred.detach()
    adversarial = make_attack(model, input_tensor, label_for_attack, attack_method, epsilon, pgd_alpha, pgd_steps)
    adv_pred, adv_conf, adv_probs = predict(model, adversarial)

    clean_rgb = denormalize(input_tensor)
    adv_rgb = denormalize(adversarial)
    defended_rgb = jpeg_compress_batch(adv_rgb, jpeg_quality) if use_jpeg else adv_rgb
    defended_norm = normalize(defended_rgb)
    def_pred, def_conf, def_probs = predict(model, defended_norm)

    image_cols = st.columns(4)
    image_cols[0].image(image.resize((192, 192), Image.Resampling.LANCZOS), caption="Original upload")
    image_cols[1].image(to_display_image(adv_rgb), caption=f"{attack_method} adversarial")
    image_cols[2].image(to_heatmap_image(clean_rgb, adv_rgb), caption="Perturbation heatmap")
    image_cols[3].image(to_display_image(defended_rgb), caption=f"JPEG Q{jpeg_quality}" if use_jpeg else "No defense")

    pred_cols = st.columns(3)
    with pred_cols[0]:
        render_prediction("Original prediction", clean_pred, clean_conf, clean_probs)
    with pred_cols[1]:
        render_prediction("Adversarial prediction", adv_pred, adv_conf, adv_probs)
    with pred_cols[2]:
        render_prediction("Defense prediction", def_pred, def_conf, def_probs)

    export = Image.new("RGB", (768, 224), "white")
    export.paste(image.resize((192, 192), Image.Resampling.LANCZOS), (0, 0))
    export.paste(to_display_image(adv_rgb), (192, 0))
    export.paste(to_heatmap_image(clean_rgb, adv_rgb), (384, 0))
    export.paste(to_display_image(defended_rgb), (576, 0))
    buffer = io.BytesIO()
    export.save(buffer, format="PNG")
    st.download_button("Download current case", buffer.getvalue(), file_name="traffic_sign_demo_case.png", mime="image/png")


if __name__ == "__main__":
    main()
