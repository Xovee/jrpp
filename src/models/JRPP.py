from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from models.gumbel_ib_filter import GumbelIBFilter, _finite_scalar, _positive_int
from models.initialization import initialize_jrpp
from models.mol_similarity import MoLSimilarity
from models.retrieval import Retrieval
from models.TransformerBlock import TransformerBlock
from utils.user_identity import configure_model_identity, validate_identity


class JRPP(nn.Module):
    """Joint retrieval and popularity prediction model."""

    def __init__(self, args, config, meta_dim: int, dropout: float = 0.2):
        super().__init__()
        user_identity = validate_identity(config.get("user_identity"))

        self.config = config
        self.emb_size = int(args.embSize)
        self.data_name = args.data_name
        self.meta_dim = int(meta_dim)
        self.input_dim = self.emb_size * 2 + self.meta_dim
        self.fusion_dim = self.emb_size + self.emb_size // 2

        retrieval_config = config["retrieval"]
        filter_config = config.get("filter", {})
        top_k = retrieval_config.get("top_k", 50)
        _positive_int("retrieval.top_k", top_k)
        keep_ratio = filter_config.get("keep_ratio", 0.8)
        _finite_scalar("filter.keep_ratio", keep_ratio)
        if keep_ratio > 1:
            raise ValueError("filter.keep_ratio must be in (0, 1].")
        for field, default in (("hidden_dim", 256), ("ib_dim", 512)):
            _positive_int("filter." + field, filter_config.get(field, default))
        _finite_scalar("filter.tau", filter_config.get("tau", 1.0))
        _finite_scalar("filter.beta", filter_config.get("beta", 1e-11), allow_zero=True)
        _finite_scalar(
            "training.ib_loss_weight",
            config.get("training", {}).get("ib_loss_weight", 1.0),
            allow_zero=True,
        )
        select_k = max(1, int(top_k * keep_ratio))
        ib_dim = int(filter_config.get("ib_dim", 512))

        self.text_projection = nn.Linear(768, self.emb_size)
        self.image_projection = nn.Linear(768, self.emb_size)

        self.mol_similarity = MoLSimilarity(retrieval_config["mol"])
        self.retrieval = Retrieval(retrieval_config, model=self)
        self.gumbel_ib_filter = GumbelIBFilter(
            cand_dim=self.input_dim,
            query_dim=self.input_dim,
            hidden_dim=int(filter_config.get("hidden_dim", 256)),
            ib_dim=ib_dim,
            select_k=select_k,
            tau=float(filter_config.get("tau", 1.0)),
            beta=float(filter_config.get("beta", 1e-11)),
            use_score=bool(filter_config.get("use_score", True)),
        )

        self.query_projection = nn.Linear(self.input_dim, self.fusion_dim)
        self.context_projection = nn.Linear(ib_dim, self.fusion_dim)
        self.joint_projection = nn.Linear(self.input_dim + ib_dim, self.fusion_dim)
        self.fusing_attn = TransformerBlock(
            input_size=self.fusion_dim, n_heads=4, attn_dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, 1),
        )

        self.reset_parameters()
        configure_model_identity(self, user_identity)

    def reset_parameters(self) -> None:
        initialize_jrpp(self)

    def _project_items(
        self,
        items_text: torch.Tensor,
        items_img: torch.Tensor,
        items_meta: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_projection(items_text)
        image = self.image_projection(items_img)
        meta = items_meta.squeeze(1)
        return torch.cat([text, image, meta], dim=-1).unsqueeze(0)

    def _project_query(
        self,
        text_vec: torch.Tensor,
        img_vec_cls: torch.Tensor,
        meta_features: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_projection(text_vec)
        image = self.image_projection(img_vec_cls)
        return torch.cat([text, image, meta_features], dim=-1)

    @staticmethod
    def _tensor_signature(tensors):
        return tuple((id(t), t._version, t.device, t.dtype) for t in tensors)

    @contextmanager
    def evaluation_cache(self, items):
        """Reuse a fixed retrieval bank within one eval session only.

        Model weights and bank tensors must remain unchanged. The cache is not
        part of state_dict and is always released on leaving this context.
        """
        if self.training:
            raise RuntimeError("Retrieval caching requires model.eval().")
        if getattr(self, "_evaluation_cache", None) is not None:
            raise RuntimeError("Nested retrieval cache contexts are not supported.")
        item_ids, text, image, meta = items
        with torch.no_grad():
            embeddings = self._project_items(text, image, meta)
            topk = self.retrieval.get_topk_related_items(embeddings, item_ids)
            topk.release_component_cache()
        self._evaluation_cache = dict(
            embeddings=embeddings,
            topk=topk,
            items=self._tensor_signature(items),
            parameters=self._tensor_signature(self.parameters()),
        )
        try:
            yield self
        finally:
            self._evaluation_cache = None

    def forward(
        self,
        text_vec,
        meta_features,
        img_vec_cls,
        user_id,
        image_id,
        items_id,
        items_text,
        items_img,
        items_meta,
    ):
        input_embeddings = self._project_query(text_vec, img_vec_cls, meta_features)
        cache = getattr(self, "_evaluation_cache", None)
        if cache is None:
            items_embeddings = self._project_items(items_text, items_img, items_meta)
            topk_model = self.retrieval.get_topk_related_items(items_embeddings, items_id)
        else:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "Cached retrieval is only valid for evaluation without gradients."
                )
            if cache["parameters"] != self._tensor_signature(self.parameters()) or cache[
                "items"
            ] != self._tensor_signature((items_id, items_text, items_img, items_meta)):
                raise RuntimeError("Model or retrieval bank changed; rebuild the evaluation cache.")
            items_embeddings, topk_model = cache["embeddings"], cache["topk"]
        top_k_scores, top_k_indices = topk_model(
            input_embeddings,
            int(self.config["retrieval"]["top_k"]),
            user_ids=user_id,
            exclude_item_ids=image_id,
        )
        top_k_embeddings = items_embeddings.squeeze(0)[top_k_indices]

        refined, _, kl_loss, _, _, row_score = self.gumbel_ib_filter(
            query_emb=input_embeddings,
            cand_emb=top_k_embeddings,
            cand_score=top_k_scores,
        )
        # Preserve each selected neighbor as a key/value token.
        query = self.query_projection(input_embeddings).unsqueeze(1)
        context = self.context_projection(refined)
        expanded_query = input_embeddings.unsqueeze(1).expand(-1, refined.size(1), -1)
        joint = self.joint_projection(torch.cat([expanded_query, refined], dim=-1))
        # Retrieval scores remain differentiable and act as an attention prior.
        prior = F.log_softmax(row_score, dim=-1).unsqueeze(1)
        output = self.fusing_attn(query, context, joint, attention_bias=prior)
        output = self.regressor(output.squeeze(1))
        return output, kl_loss
