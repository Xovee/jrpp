from typing import Callable, List

import torch
import torch.nn.functional as F


class RecoMoLQueryEmbeddingsFn(torch.nn.Module):
    def __init__(
        self,
        query_embedding_dim: int,
        query_dot_product_groups: int,
        dot_product_dimension: int,
        dot_product_l2_norm: bool,
        proj_fn: Callable[[int, int], torch.nn.Module],
        eps: float,
        uid_embedding_hash_sizes: List[int],
        uid_dropout_rate: float,
        uid_embedding_level_dropout: bool = False,
    ) -> None:
        super().__init__()
        self._uid_embedding_hash_sizes: List[int] = uid_embedding_hash_sizes
        self._query_emb_based_dot_product_groups: int = query_dot_product_groups - len(
            self._uid_embedding_hash_sizes
        )
        self._query_emb_proj_module: torch.nn.Module = proj_fn(
            query_embedding_dim,
            dot_product_dimension * self._query_emb_based_dot_product_groups,
        )
        self._dot_product_dimension: int = dot_product_dimension
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        if len(self._uid_embedding_hash_sizes) > 0:
            for i, hash_size in enumerate(self._uid_embedding_hash_sizes):
                embedding = torch.nn.Embedding(hash_size + 1, dot_product_dimension, padding_idx=0)
                setattr(
                    self,
                    f"_uid_embeddings_{i}",
                    embedding,
                )
        self._uid_dropout_rate: float = uid_dropout_rate
        self._uid_embedding_level_dropout: bool = uid_embedding_level_dropout
        self._eps: float = float(eps)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        split_query_embeddings = self._query_emb_proj_module(input_embeddings).reshape(
            (
                input_embeddings.size(0),
                self._query_emb_based_dot_product_groups,
                self._dot_product_dimension,
            )
        )

        if len(self._uid_embedding_hash_sizes) > 0:
            all_uid_embeddings = []
            uid = kwargs["user_ids"]
            for i, hash_size in enumerate(self._uid_embedding_hash_sizes):
                uid_embeddings = getattr(self, f"_uid_embeddings_{i}")(uid)
                if self._uid_dropout_rate > 0.0:
                    if self._uid_embedding_level_dropout:
                        if self.training:
                            uid_dropout_mask = (
                                torch.rand(uid_embeddings.size()[:-1], device=uid_embeddings.device)
                                > self._uid_dropout_rate
                            )
                            uid_embeddings = (
                                uid_embeddings
                                * uid_dropout_mask.unsqueeze(-1)
                                / (1.0 - self._uid_dropout_rate)
                            )
                    else:
                        uid_embeddings = F.dropout(
                            uid_embeddings, p=self._uid_dropout_rate, training=self.training
                        )
                all_uid_embeddings.append(uid_embeddings.unsqueeze(1))
            split_query_embeddings = torch.cat([split_query_embeddings] + all_uid_embeddings, dim=1)

        if self._dot_product_l2_norm:
            split_query_embeddings = split_query_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_query_embeddings,
                    ord=None,
                    dim=-1,
                    keepdim=True,
                ),
                min=self._eps,
            )
        return split_query_embeddings
