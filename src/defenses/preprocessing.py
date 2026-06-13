import io

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def jpeg_compress_batch(images: torch.Tensor, quality: int) -> torch.Tensor:
    output = []
    for image in images.detach().cpu():
        arr = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        output.append(torch.from_numpy(np.array(compressed)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(output, dim=0).to(images.device)


def median_filter_batch(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    output = []
    for image in images.detach().cpu():
        arr = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        filtered = cv2.medianBlur(arr, int(kernel_size))
        output.append(torch.from_numpy(filtered).permute(2, 0, 1).float() / 255.0)
    return torch.stack(output, dim=0).to(images.device)


def preprocess_pixels(images: torch.Tensor, method: str, params: dict) -> torch.Tensor:
    if method == "gaussian_blur":
        kernel_size = int(params.get("kernel_size", 3))
        sigma = float(params.get("sigma", 0.6))
        return TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    if method == "median_filter":
        return median_filter_batch(images, int(params.get("kernel_size", 3)))
    if method == "jpeg_compression":
        return jpeg_compress_batch(images, int(params.get("quality", 75)))
    raise ValueError(f"Unsupported defense method: {method}")
