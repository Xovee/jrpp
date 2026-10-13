from typing import Tuple

import torch

from models.mol_similarity import MoLSimilarity


class MoLAvgTopK(torch.nn.Module):
    """Shortlist by mean component similarity, then rerank with the full MoL."""

    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        avg_top_k: int,
    ) -> None:
        super().__init__()
        self._mol_module = mol_module
        self._item_embeddings = item_embeddings.squeeze(0)
        self._item_ids = item_ids.squeeze(0)
        self._mol_item_embeddings = mol_module.get_item_component_embeddings(
            self._item_embeddings
        ).permute(1, 0, 2)
        components = self._mol_item_embeddings.size(0)
        self._avg_mol_item_embeddings_t = (self._mol_item_embeddings.sum(0) / components).transpose(
            0, 1
        )
        self._avg_top_k = avg_top_k

    @property
    def mol_module(self) -> MoLSimilarity:
        return self._mol_module

    def release_component_cache(self):
        """After eval index construction, only the component mean is needed."""
        self._mol_item_embeddings = None

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        exclude_item_ids: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = query_embeddings.size()
        mol_query_embeddings = self.mol_module.get_query_component_embeddings(
            query_embeddings, **kwargs
        )
        P_Q = mol_query_embeddings.size(1)
        N = self._item_embeddings.size(0)
        if k <= 0 or self._avg_top_k <= 0 or N == 0:
            raise ValueError("Retrieval requires positive k and a non-empty candidate bank.")
        candidate_k = min(self._avg_top_k, N)

        avg_sim_values = torch.mm(
            mol_query_embeddings.sum(1) / P_Q, self._avg_mol_item_embeddings_t
        )
        excluded = None
        item_ids = self._item_ids.reshape(1, -1).to(avg_sim_values.device)
        if exclude_item_ids is not None:
            excluded = exclude_item_ids.reshape(-1, 1).to(
                device=avg_sim_values.device, dtype=self._item_ids.dtype
            )
            excluded_mask = item_ids.eq(excluded)
            # A rectangular batch must use a candidate count valid for every row.
            candidate_k = min(candidate_k, int((~excluded_mask).sum(1).min().item()))
            avg_sim_values = avg_sim_values.masked_fill(
                excluded_mask, torch.finfo(avg_sim_values.dtype).min
            )

        if candidate_k == 0:
            raise ValueError("No retrieval candidates remain after excluding the query itself.")
        output_k = min(k, candidate_k)
        _, avg_sim_top_k_indices = torch.topk(avg_sim_values, k=candidate_k, dim=1)

        avg_filtered_item_embeddings = self._item_embeddings[avg_sim_top_k_indices].reshape(
            B, candidate_k, D
        )
        candidate_scores = self.mol_module(query_embeddings, avg_filtered_item_embeddings, **kwargs)
        if excluded is not None:
            avg_filtered_item_ids = self._item_ids[avg_sim_top_k_indices].to(
                candidate_scores.device
            )
            candidate_scores = candidate_scores.masked_fill(
                avg_filtered_item_ids.eq(excluded.to(candidate_scores.device)),
                torch.finfo(candidate_scores.dtype).min,
            )
        top_k_logits, top_k_indices = torch.topk(
            input=candidate_scores, k=output_k, dim=1, largest=True, sorted=sorted
        )
        top_k_item_indices = torch.gather(avg_sim_top_k_indices, dim=1, index=top_k_indices)
        return top_k_logits, top_k_item_indices
