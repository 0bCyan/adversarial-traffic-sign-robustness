import torch

from src.defenses.preprocessing import jpeg_compress_batch, preprocess_pixels
from src.experiment_utils import denormalize, normalize


def test_normalize_denormalize_round_trip() -> None:
    images = torch.rand(2, 3, 16, 16)
    recovered = denormalize(normalize(images))
    assert torch.allclose(images, recovered, atol=1e-6)


def test_jpeg_compress_batch_preserves_shape_and_range() -> None:
    images = torch.rand(2, 3, 16, 16)
    compressed = jpeg_compress_batch(images, quality=75)
    assert compressed.shape == images.shape
    assert float(compressed.min()) >= 0.0
    assert float(compressed.max()) <= 1.0


def test_preprocess_pixels_dispatches_jpeg() -> None:
    images = torch.rand(1, 3, 16, 16)
    defended = preprocess_pixels(images, "jpeg_compression", {"quality": 50})
    assert defended.shape == images.shape
