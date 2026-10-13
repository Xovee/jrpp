import math
from numbers import Integral, Real
from typing import List, Optional

import torch

from models.mol.mol import (
    GeGLU,
    IdentityMLPProjectionFn,
    MoLSimilarity,
    SoftmaxDropoutCombiner,
    SwiGLU,
    init_mlp_xavier_weights_zero_bias,
)
from models.mol.mol_query_embeddings import RecoMoLQueryEmbeddingsFn


def create_mol_interaction_module(
    query_embedding_dim: int,
    item_embedding_dim: int,
    dot_product_dimension: int,
    query_dot_product_groups: int,
    item_dot_product_groups: int,
    temperature: float,
    query_use_identity_fn: bool,
    query_dropout_rate: float,
    query_hidden_dim: int,
    item_use_identity_fn: bool,
    item_dropout_rate: float,
    item_hidden_dim: int,
    gating_query_hidden_dim: int,
    gating_qi_hidden_dim: int,
    gating_item_hidden_dim: int,
    softmax_dropout_rate: float,
    bf16_training: bool,
    gating_query_fn: bool = True,
    gating_item_fn: bool = True,
    dot_product_l2_norm: bool = True,
    query_nonlinearity: str = "geglu",
    item_nonlinearity: str = "geglu",
    uid_dropout_rate: float = 0.5,
    uid_embedding_hash_sizes: Optional[List[int]] = None,
    uid_embedding_level_dropout: bool = False,
    gating_combination_type: str = "glu_silu",
    gating_item_dropout_rate: float = 0.0,
    gating_qi_dropout_rate: float = 0.0,
    eps: float = 1e-6,
    normalize_gate_logits: bool = False,
) -> MoLSimilarity:
    for name, value in (
        ("query_embedding_dim", query_embedding_dim),
        ("item_embedding_dim", item_embedding_dim),
        ("dot_product_dimension", dot_product_dimension),
        ("query_dot_product_groups", query_dot_product_groups),
        ("item_dot_product_groups", item_dot_product_groups),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"MoL {name} must be a positive integer.")
    for name, value in (("temperature", temperature), ("eps", eps)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"MoL {name} must be finite and positive.")
    for value in (
        query_dropout_rate,
        item_dropout_rate,
        softmax_dropout_rate,
        uid_dropout_rate,
        gating_item_dropout_rate,
        gating_qi_dropout_rate,
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 1:
            raise ValueError("MoL dropout rates must be in [0, 1).")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in (uid_embedding_hash_sizes or [])
    ):
        raise ValueError("User embedding hash sizes must be positive integers.")
    if gating_combination_type not in ("glu_silu", "glu_silu_ln", "silu", "none"):
        raise ValueError("Unknown MoL gating combination type.")
    # Reshape content components before appending user-ID components.
    content_groups = query_dot_product_groups - len(uid_embedding_hash_sizes or [])
    if content_groups < 1:
        raise ValueError("query_dot_product_groups must exceed the number of user hash embeddings.")
    if (
        query_use_identity_fn
        and (dot_product_dimension * content_groups) % query_dot_product_groups
    ):
        raise ValueError(
            "Identity query projection requires an evenly divisible flat projection width."
        )
    if gating_combination_type in ("glu_silu", "glu_silu_ln") and not (
        gating_query_fn and gating_item_fn
    ):
        raise ValueError("GLU gating requires both query and item branches.")
    if query_nonlinearity not in ("geglu", "swiglu") or item_nonlinearity not in (
        "geglu",
        "swiglu",
    ):
        raise ValueError("MoL nonlinearity must be geglu or swiglu.")
    mol_module = MoLSimilarity(
        query_embedding_dim=query_embedding_dim,
        item_embedding_dim=item_embedding_dim,
        dot_product_dimension=dot_product_dimension,
        query_dot_product_groups=query_dot_product_groups,
        item_dot_product_groups=item_dot_product_groups,
        temperature=temperature,
        dot_product_l2_norm=dot_product_l2_norm,
        query_embeddings_fn=RecoMoLQueryEmbeddingsFn(
            query_embedding_dim=query_embedding_dim,
            query_dot_product_groups=query_dot_product_groups,
            dot_product_dimension=dot_product_dimension,
            dot_product_l2_norm=dot_product_l2_norm,
            proj_fn=lambda input_dim, output_dim: (
                IdentityMLPProjectionFn(
                    input_dim=input_dim,
                    output_num_features=query_dot_product_groups,
                    output_dim=output_dim // query_dot_product_groups,
                    input_dropout_rate=query_dropout_rate,
                )
                if query_use_identity_fn
                else (
                    torch.nn.Sequential(
                        torch.nn.Dropout(p=query_dropout_rate),
                        GeGLU(
                            in_features=input_dim,
                            out_features=query_hidden_dim,
                        )
                        if query_nonlinearity == "geglu"
                        else SwiGLU(
                            in_features=input_dim,
                            out_features=query_hidden_dim,
                        ),
                        torch.nn.Linear(
                            in_features=query_hidden_dim,
                            out_features=output_dim,
                        ),
                    )
                    if query_hidden_dim > 0
                    else torch.nn.Sequential(
                        torch.nn.Dropout(p=query_dropout_rate),
                        torch.nn.Linear(
                            in_features=input_dim,
                            out_features=output_dim,
                        ),
                    )
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
            uid_embedding_hash_sizes=uid_embedding_hash_sizes or [],
            uid_dropout_rate=uid_dropout_rate,
            uid_embedding_level_dropout=uid_embedding_level_dropout,
            eps=eps,
        ),
        item_proj_fn=lambda input_dim, output_dim: (
            IdentityMLPProjectionFn(
                input_dim=input_dim,
                output_num_features=item_dot_product_groups,
                output_dim=output_dim // item_dot_product_groups,
                input_dropout_rate=item_dropout_rate,
            )
            if item_use_identity_fn
            else (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    GeGLU(
                        in_features=input_dim,
                        out_features=item_hidden_dim,
                    )
                    if item_nonlinearity == "geglu"
                    else SwiGLU(in_features=input_dim, out_features=item_hidden_dim),
                    torch.nn.Linear(
                        in_features=item_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
                if item_hidden_dim > 0
                else torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    torch.nn.Linear(
                        in_features=input_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            )
        ),
        gating_query_only_partial_fn=lambda input_dim, output_dim: (
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=gating_query_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=gating_query_hidden_dim,
                    out_features=output_dim,
                    bias=False,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if gating_query_fn
            else None
        ),
        gating_item_only_partial_fn=lambda input_dim, output_dim: (
            torch.nn.Sequential(
                torch.nn.Dropout(p=gating_item_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=gating_item_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=gating_item_hidden_dim,
                    out_features=output_dim,
                    bias=False,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if gating_item_fn
            else None
        ),
        gating_qi_partial_fn=lambda input_dim, output_dim: (
            torch.nn.Sequential(
                torch.nn.Dropout(p=gating_qi_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=gating_qi_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=gating_qi_hidden_dim,
                    out_features=output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if gating_qi_hidden_dim > 0
            else torch.nn.Sequential(
                torch.nn.Dropout(p=gating_qi_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
        ),
        gating_combination_type=gating_combination_type,
        gating_normalization_fn=lambda _: SoftmaxDropoutCombiner(
            dropout_rate=softmax_dropout_rate,
            eps=1e-6,
            normalize_gate_logits=normalize_gate_logits,
        ),
        eps=eps,
        bf16_training=bf16_training,
    )
    return mol_module
