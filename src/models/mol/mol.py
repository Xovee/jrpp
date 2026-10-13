from typing import Callable, Optional

import torch
import torch.nn.functional as F


def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


def _softmax_dropout_combiner_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    dropout_pr: float,
    eps: float,
    training: bool,
) -> torch.Tensor:
    """Combine expert scores using normalized gates and training-time dropout."""
    x = F.softmax(x, dim=-1)

    if dropout_pr > 0.0:
        x = F.dropout(x, p=dropout_pr, training=training)
        # Reweight surviving experts; a fully dropped row contributes zero.
        x = x / torch.clamp(x.sum(-1, keepdim=True), min=eps)

    return (x * y).sum(-1)


class SoftmaxDropoutCombiner(torch.nn.Module):
    def __init__(
        self,
        dropout_rate: float,
        eps: float,
        normalize_gate_logits: bool = False,
    ) -> None:
        super().__init__()

        self._dropout_rate: float = dropout_rate
        self._eps: float = eps
        self._normalize_gate_logits = normalize_gate_logits

    def forward(
        self,
        gating_weights: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self._normalize_gate_logits:
            # Normalize gate logits in FP32 without affine parameters.
            gating_weights = F.layer_norm(
                gating_weights.float(),
                (gating_weights.shape[-1],),
                eps=self._eps,
            )
        combined_logits = _softmax_dropout_combiner_fn(
            x=gating_weights,
            y=x,
            dropout_pr=self._dropout_rate,
            eps=self._eps,
            training=self.training,
        )

        return combined_logits


class IdentityMLPProjectionFn(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_num_features: int,
        output_dim: int,
        input_dropout_rate: float,
    ) -> None:
        super().__init__()

        if input_dim < output_dim:
            raise ValueError(
                "Identity projection requires input_dim >= output_dim; increase --embedding-size or disable the identity projection."
            )
        if output_num_features < 1:
            raise ValueError("Identity projection requires at least one component.")
        self._output_num_features = output_num_features
        self._output_dim = output_dim
        if output_num_features > 1:
            self._proj_mlp = torch.nn.Sequential(
                torch.nn.Dropout(p=input_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=(output_num_features - 1) * output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_emb_0 = x[..., : self._output_dim]
        if self._output_num_features > 1:
            return torch.cat([output_emb_0, self._proj_mlp(x)], dim=-1)
        return output_emb_0


class GeGLU(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._w = torch.nn.Parameter(
            torch.empty((in_features, out_features * 2)).normal_(mean=0, std=0.02),
        )
        self._b = torch.nn.Parameter(
            torch.zeros(
                (
                    1,
                    out_features * 2,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[:-1]
        lhs, rhs = torch.split(
            torch.mm(x.reshape(-1, self._in_features), self._w) + self._b,
            [self._out_features, self._out_features],
            dim=-1,
        )
        return (F.gelu(lhs) * rhs).reshape(bs + (self._out_features,))


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._w = torch.nn.Parameter(
            torch.empty((in_features, out_features * 2)).normal_(mean=0, std=0.02),
        )
        self._b = torch.nn.Parameter(
            torch.zeros(
                (
                    1,
                    out_features * 2,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[:-1]
        lhs, rhs = torch.split(
            torch.mm(x.reshape(-1, self._in_features), self._w) + self._b,
            [self._out_features, self._out_features],
            dim=-1,
        )
        return (F.silu(lhs) * rhs).reshape(bs + (self._out_features,))


class MoLGatingFn(torch.nn.Module):
    def __init__(
        self,
        num_logits: int,
        query_embedding_dim: int,
        item_embedding_dim: int,
        query_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        item_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        qi_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        combination_type: str,
        normalization_fn: Callable[[int], torch.nn.Module],
    ) -> None:
        super().__init__()
        self._query_only_partial_module: Optional[torch.nn.Module] = (
            query_only_partial_fn(query_embedding_dim, num_logits)
            if query_only_partial_fn
            else None
        )

        self._item_only_partial_module: Optional[torch.nn.Module] = (
            item_only_partial_fn(item_embedding_dim, num_logits) if item_only_partial_fn else None
        )

        self._qi_partial_module: Optional[torch.nn.Module] = (
            qi_partial_fn(
                num_logits,
                num_logits,
            )
            if qi_partial_fn is not None
            else None
        )
        if (
            self._query_only_partial_module is None
            and self._item_only_partial_module is None
            and self._qi_partial_module is None
        ):
            raise ValueError(
                "At least one of query_only_partial_fn, item_only_partial_fn, "
                "and qi_partial_fn must not be None."
            )
        self._num_logits: int = num_logits
        self._combination_type: str = combination_type
        self._normalization_fn: torch.nn.Module = normalization_fn(num_logits)

    def forward(
        self,
        logits: torch.Tensor,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Combine query, item and expert logits into [batch, candidates] scores."""
        B, X, _ = logits.size()
        query_partial_inputs, item_partial_inputs, qi_partial_inputs = None, None, None
        if self._query_only_partial_module is not None:
            query_partial_inputs = self._query_only_partial_module(query_embeddings).unsqueeze(1)
        if self._item_only_partial_module is not None:
            item_partial_inputs = self._item_only_partial_module(item_embeddings)
        if self._qi_partial_module is not None:
            qi_partial_inputs = self._qi_partial_module(logits)

        if self._combination_type == "glu_silu":
            gating_inputs = query_partial_inputs * item_partial_inputs + qi_partial_inputs
            gating_weights = gating_inputs * F.sigmoid(gating_inputs)
        elif self._combination_type == "glu_silu_ln":
            gating_inputs = query_partial_inputs * item_partial_inputs + qi_partial_inputs
            gating_weights = gating_inputs * F.sigmoid(
                F.layer_norm(gating_inputs, normalized_shape=[self._num_logits])
            )
        elif self._combination_type == "silu":
            if query_partial_inputs is not None:
                gating_inputs = query_partial_inputs.expand(-1, X, -1)
            else:
                gating_inputs = None

            if gating_inputs is None:
                gating_inputs = item_partial_inputs
            elif item_partial_inputs is not None:
                gating_inputs = gating_inputs + item_partial_inputs

            if gating_inputs is None:
                gating_inputs = qi_partial_inputs
            elif qi_partial_inputs is not None:
                gating_inputs = gating_inputs + qi_partial_inputs

            gating_weights = gating_inputs * F.sigmoid(gating_inputs)
        elif self._combination_type == "none":
            gating_inputs = query_partial_inputs
            if gating_inputs is None:
                gating_inputs = item_partial_inputs
            elif item_partial_inputs is not None:
                gating_inputs = gating_inputs + item_partial_inputs
            if gating_inputs is None:
                gating_inputs = qi_partial_inputs
            elif qi_partial_inputs is not None:
                gating_inputs = gating_inputs + qi_partial_inputs
            gating_weights = gating_inputs
        else:
            raise ValueError(f"Unknown combination_type {self._combination_type}")
        return self._normalization_fn(gating_weights, logits)


class MoLSimilarity(torch.nn.Module):
    def __init__(
        self,
        query_embedding_dim: int,
        item_embedding_dim: int,
        dot_product_dimension: int,
        query_dot_product_groups: int,
        item_dot_product_groups: int,
        temperature: float,
        dot_product_l2_norm: bool,
        query_embeddings_fn: torch.nn.Module,
        item_proj_fn: Callable[[int, int], torch.nn.Module],
        gating_query_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        gating_item_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        gating_qi_partial_fn: Optional[Callable[[int], torch.nn.Module]],
        gating_combination_type: str,
        gating_normalization_fn: Callable[[int], torch.nn.Module],
        eps: float,
        bf16_training: bool = False,
    ) -> None:
        super().__init__()

        self._gating_fn: MoLGatingFn = MoLGatingFn(
            num_logits=query_dot_product_groups * item_dot_product_groups,
            query_embedding_dim=query_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            query_only_partial_fn=gating_query_only_partial_fn,
            item_only_partial_fn=gating_item_only_partial_fn,
            qi_partial_fn=gating_qi_partial_fn,
            combination_type=gating_combination_type,
            normalization_fn=gating_normalization_fn,
        )
        self._query_embeddings_fn: torch.nn.Module = query_embeddings_fn
        self._item_proj_module: torch.nn.Module = item_proj_fn(
            item_embedding_dim,
            dot_product_dimension * item_dot_product_groups,
        )
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        self._query_dot_product_groups: int = query_dot_product_groups
        self._item_dot_product_groups: int = item_dot_product_groups
        self._dot_product_dimension: int = dot_product_dimension
        self._temperature: float = temperature
        self._eps: float = eps
        self._bf16_training: bool = bf16_training

    def get_query_component_embeddings(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self._query_embeddings_fn(input_embeddings, **kwargs)

    def get_item_component_embeddings(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        split_item_embeddings = self._item_proj_module(input_embeddings).reshape(
            input_embeddings.size()[:-1]
            + (
                self._item_dot_product_groups,
                self._dot_product_dimension,
            )
        )
        if self._dot_product_l2_norm:
            split_item_embeddings = split_item_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_item_embeddings,
                    ord=None,
                    dim=-1,
                    keepdim=True,
                ),
                min=self._eps,
            )
        return split_item_embeddings

    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        autocast_enabled = self._bf16_training and input_embeddings.is_cuda
        with torch.autocast(
            enabled=autocast_enabled,
            dtype=torch.bfloat16,
            device_type=input_embeddings.device.type,
        ):
            B = input_embeddings.size(0)
            B_prime, X, D = item_embeddings.shape
            split_user_embeddings = self.get_query_component_embeddings(input_embeddings, **kwargs)
            split_item_embeddings = self.get_item_component_embeddings(item_embeddings, **kwargs)
            if B_prime == 1:
                logits = torch.einsum(
                    "bnd,xmd->bxnm", split_user_embeddings, split_item_embeddings.squeeze(0)
                ).reshape(B, X, self._query_dot_product_groups * self._item_dot_product_groups)
            else:
                logits = torch.einsum(
                    "bnd,bxmd->bxnm", split_user_embeddings, split_item_embeddings
                ).reshape(B, X, self._query_dot_product_groups * self._item_dot_product_groups)

            gating_outputs = self._gating_fn(
                logits=logits / self._temperature,
                query_embeddings=input_embeddings,
                item_embeddings=item_embeddings,
            )

            return gating_outputs
