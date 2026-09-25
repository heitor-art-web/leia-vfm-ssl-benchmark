import pytest
import torch
from torch import nn

from leia_benchmark.ssl.mean_teacher import (
    confidence_mask,
    consistency_mse,
    create_ema_teacher,
    update_ema,
)


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2),
    )


def test_create_teacher_is_frozen_and_eval():
    student = _model()
    teacher = create_ema_teacher(student)
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())


def test_update_ema_updates_parameters_and_batchnorm_buffers():
    student = _model()
    teacher = create_ema_teacher(student)

    with torch.no_grad():
        student[0].weight.fill_(4.0)
        teacher[0].weight.fill_(2.0)
        student[1].running_mean.fill_(6.0)
        teacher[1].running_mean.fill_(2.0)
        student[1].num_batches_tracked.fill_(7)
        teacher[1].num_batches_tracked.fill_(1)

    update_ema(student, teacher, decay=0.5)

    assert torch.allclose(teacher[0].weight, torch.full_like(teacher[0].weight, 3.0))
    assert torch.allclose(
        teacher[1].running_mean,
        torch.full_like(teacher[1].running_mean, 4.0),
    )
    assert int(teacher[1].num_batches_tracked.item()) == 7
    assert not teacher.training


def test_update_ema_validates_decay():
    student = _model()
    teacher = create_ema_teacher(student)
    with pytest.raises(ValueError):
        update_ema(student, teacher, decay=1.1)


def test_confidence_mask_validates_threshold():
    logits = torch.tensor([[[[3.0]], [[0.0]]]])
    mask = confidence_mask(logits, threshold=0.9)
    assert bool(mask.item())
    with pytest.raises(ValueError):
        confidence_mask(logits, threshold=-0.1)


def test_consistency_mse_handles_empty_mask_without_nan():
    student = torch.zeros((1, 2, 2, 2))
    teacher = torch.ones((1, 2, 2, 2))
    mask = torch.zeros((1, 2, 2), dtype=torch.bool)
    loss = consistency_mse(student, teacher, mask)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_consistency_mse_validates_shapes():
    student = torch.zeros((1, 2, 2, 2))
    teacher = torch.zeros((1, 2, 3, 2))
    with pytest.raises(ValueError):
        consistency_mse(student, teacher)
