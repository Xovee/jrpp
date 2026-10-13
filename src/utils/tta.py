"""Prediction averaging with modality-relative, antithetic feature noise."""

import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F


def relative_noise(features, scale, generator=None):
    """Noise has L2 norm scale * ||features|| per row; zero inputs stay zero."""
    direction = torch.randn(
        features.shape, device=features.device, dtype=features.dtype, generator=generator
    )
    return F.normalize(direction, dim=-1) * features.norm(dim=-1, keepdim=True) * scale


def validate_tta_settings(runs, relative_scale, anchor_weight):
    if isinstance(runs, bool) or not isinstance(runs, Integral) or runs < 0 or runs % 2:
        raise ValueError("Paired TTA runs must be a nonnegative even integer.")
    if (
        isinstance(relative_scale, bool)
        or not isinstance(relative_scale, Real)
        or not math.isfinite(relative_scale)
        or relative_scale < 0
    ):
        raise ValueError("TTA relative scale must be finite and nonnegative.")
    if (
        isinstance(anchor_weight, bool)
        or not isinstance(anchor_weight, Real)
        or not 0 <= anchor_weight <= 1
    ):
        raise ValueError("TTA anchor weight must be in [0, 1].")


@torch.no_grad()
def predict_tta(
    model, batch, items, runs=4, relative_scale=0.02, anchor_weight=0.5, generator=None
):
    validate_tta_settings(runs, relative_scale, anchor_weight)
    if model.training:
        raise ValueError("TTA requires model.eval().")

    def predict(text, image):
        return model(text, batch.meta_features, image, batch.user_id, batch.image_id, *items)[
            0
        ].reshape(-1)

    original = predict(batch.text_vec, batch.img_vec_cls)
    if runs == 0 or relative_scale == 0 or anchor_weight == 1:
        return original
    total = torch.zeros_like(original)
    for _ in range(runs // 2):
        text_noise = relative_noise(batch.text_vec, relative_scale, generator)
        image_noise = relative_noise(batch.img_vec_cls, relative_scale, generator)
        total += predict(batch.text_vec + text_noise, batch.img_vec_cls + image_noise)
        total += predict(batch.text_vec - text_noise, batch.img_vec_cls - image_noise)
    return anchor_weight * original + (1 - anchor_weight) * total / runs
