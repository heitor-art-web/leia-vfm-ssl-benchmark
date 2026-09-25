from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def create_ema_teacher(student: nn.Module) -> nn.Module:
    """Create a frozen evaluation-mode copy of the student."""
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """EMA-update parameters *and buffers* from student into teacher.

    Updating only parameters leaves BatchNorm running statistics stale, which is
    incorrect for architectures containing stateful buffers. Floating tensors
    are EMA-updated; integer/counter buffers are copied exactly.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")

    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    if student_state.keys() != teacher_state.keys():
        raise ValueError("student and teacher state_dict keys do not match")

    for name, teacher_tensor in teacher_state.items():
        student_tensor = student_state[name].detach().to(
            device=teacher_tensor.device,
            dtype=teacher_tensor.dtype,
        )
        if torch.is_floating_point(teacher_tensor) or torch.is_complex(teacher_tensor):
            teacher_tensor.mul_(decay).add_(student_tensor, alpha=1.0 - decay)
        else:
            teacher_tensor.copy_(student_tensor)

    teacher.eval()


def confidence_mask(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    probs = torch.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    return conf >= threshold


def consistency_mse(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have identical shape")

    student_p = torch.softmax(student_logits, dim=1)
    teacher_p = torch.softmax(teacher_logits.detach(), dim=1)
    loss = (student_p - teacher_p).pow(2).mean(dim=1)

    if mask is None:
        return loss.mean()
    if mask.shape != loss.shape:
        raise ValueError("mask shape must equal logits shape without the class dimension")
    if not torch.any(mask):
        return loss.sum() * 0.0
    return loss[mask].mean()
