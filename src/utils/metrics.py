from typing import Dict

import numpy as np
import scipy.stats


def regression_metrics(predictions, targets) -> Dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {target.shape}")

    if not pred.size or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require nonempty, finite predictions and targets.")

    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    src, _ = scipy.stats.spearmanr(target, pred)
    if not np.isfinite(src):
        src = 0.0
    return {"mse": mse, "mae": mae, "src": float(src)}
