from __future__ import annotations

import copy
import torch
from torch import nn


@torch.no_grad()
def create_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    for s, t in zip(student.parameters(), teacher.parameters(), strict=True):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)


def confidence_mask(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    return conf >= threshold


def consistency_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    student_p = torch.softmax(student_logits, dim=1)
    teacher_p = torch.softmax(teacher_logits.detach(), dim=1)
    loss = (student_p - teacher_p).pow(2).mean(dim=1)
    if mask is None:
        return loss.mean()
    if not torch.any(mask):
        return loss.sum() * 0.0
    return loss[mask].mean()
