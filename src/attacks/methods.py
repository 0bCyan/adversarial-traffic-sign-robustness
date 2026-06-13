import torch
import torch.nn.functional as F


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    data_min: torch.Tensor,
    data_max: torch.Tensor,
) -> torch.Tensor:
    images = images.detach().clone().requires_grad_(True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    adversarial = images + epsilon * images.grad.sign()
    return torch.max(torch.min(adversarial, data_max), data_min).detach()


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    data_min: torch.Tensor,
    data_max: torch.Tensor,
) -> torch.Tensor:
    original = images.detach()
    adversarial = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    adversarial = torch.max(torch.min(adversarial, data_max), data_min).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)
        logits = model(adversarial)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adversarial = adversarial + alpha * adversarial.grad.sign()
        delta = torch.clamp(adversarial - original, min=-epsilon, max=epsilon)
        adversarial = torch.max(torch.min(original + delta, data_max), data_min).detach()
    return adversarial

