"""Parameter initialization for JRPP."""

import copy

import numpy as np
import torch
from torch import nn

from models.mol.mol import GeGLU, SwiGLU
from models.TransformerBlock import TransformerBlock

OUTPUT_BIAS_POLICY = "train-mean-output-bias-v1"


def initialize_jrpp(model: nn.Module) -> None:
    """Use Xavier projections, unit normalization scales and small embeddings."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, TransformerBlock):
            for weight in (module.W_q, module.W_k, module.W_v, module.W_o):
                nn.init.xavier_uniform_(weight)
        elif isinstance(module, (GeGLU, SwiGLU)):
            # Initialize the two GLU projections independently.
            for weight in module._w.chunk(2, dim=-1):
                nn.init.xavier_uniform_(weight)
            nn.init.zeros_(module._b)


def initialize_output_bias_from_train(model: nn.Module, train_split, data_protocol) -> None:
    """Set only the final regressor bias from complete training labels."""
    count = len(train_split)
    if count == 0:
        raise ValueError("Output-bias initialization requires nonempty train labels.")
    if hasattr(train_split, "labels"):
        labels = train_split.labels.detach().cpu().numpy()
    else:
        labels = [float(sample.y.item()) for sample in train_split]
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size != count or not np.isfinite(values).all():
        raise ValueError("Output-bias initialization requires one finite label per train sample.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values, dtype=np.float64))
    if not np.isfinite(mean):
        raise ValueError("Train-label float64 mean must be finite.")
    output = model.regressor[-1]
    if not isinstance(output, nn.Linear) or output.bias is None or output.bias.numel() != 1:
        raise ValueError("Expected the final scalar Linear regressor with a bias.")
    rounded = torch.tensor(mean, dtype=output.bias.dtype, device=output.bias.device)
    if not torch.isfinite(rounded):
        raise ValueError("Train-label mean is not finite in the regressor parameter dtype.")
    with torch.no_grad():
        output.bias.fill_(mean)
    model.initialization_policy = OUTPUT_BIAS_POLICY
    model.output_bias_initialization = {
        "train_count": count,
        "train_mean_float64": mean,
        "actual_bias": float(output.bias.detach().cpu().item()),
        "parameter_dtype": str(output.bias.dtype),
        "data_protocol": copy.deepcopy(data_protocol),
    }
