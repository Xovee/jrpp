import re

from models.mol.mol_top_k import MoLAvgTopK


class Retrieval:
    def __init__(self, config, model):
        method = config["mol"]["top_k_method"]
        match = re.fullmatch(r"MoLAvgTopK(\d+)", method)
        if not match or int(match.group(1)) < 1:
            raise ValueError(f"Invalid top-k method {method!r}. Expected e.g. MoLAvgTopK100.")
        self.avg_top_k = int(match.group(1))
        self.model = model

    def get_topk_related_items(self, item_embeddings, item_ids):
        return MoLAvgTopK(
            mol_module=self.model.mol_similarity,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=self.avg_top_k,
        )
