import torch.nn as nn

from models.mol.mol_utils import create_mol_interaction_module


class MoLSimilarity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mol_module = create_mol_interaction_module(
            query_embedding_dim=config["query_embedding_dim"],
            item_embedding_dim=config["item_embedding_dim"],
            dot_product_dimension=config["dot_product_dimension"],
            query_dot_product_groups=config["query_dot_product_groups"],
            item_dot_product_groups=config["item_dot_product_groups"],
            temperature=config["temperature"],
            query_use_identity_fn=config["query_use_identity_fn"],
            query_dropout_rate=config["query_dropout_rate"],
            query_hidden_dim=config["query_hidden_dim"],
            item_use_identity_fn=config["item_use_identity_fn"],
            item_dropout_rate=config["item_dropout_rate"],
            item_hidden_dim=config["item_hidden_dim"],
            gating_query_hidden_dim=config["gating_query_hidden_dim"],
            gating_qi_hidden_dim=config["gating_qi_hidden_dim"],
            gating_item_hidden_dim=config["gating_item_hidden_dim"],
            softmax_dropout_rate=config["softmax_dropout_rate"],
            bf16_training=config["bf16_training"],
            gating_query_fn=config["gating_query_fn"],
            gating_item_fn=config["gating_item_fn"],
            dot_product_l2_norm=config["dot_product_l2_norm"],
            query_nonlinearity=config["query_nonlinearity"],
            item_nonlinearity=config["item_nonlinearity"],
            uid_dropout_rate=config["uid_dropout_rate"],
            uid_embedding_hash_sizes=config["uid_embedding_hash_sizes"],
            uid_embedding_level_dropout=config["uid_embedding_level_dropout"],
            gating_combination_type=config["gating_combination_type"],
            gating_item_dropout_rate=config["gating_item_dropout_rate"],
            gating_qi_dropout_rate=config["gating_qi_dropout_rate"],
            eps=config["eps"],
            normalize_gate_logits=config.get("normalize_gate_logits", False),
        )

    def forward(self, input_embeddings, item_embeddings, user_ids):

        return self.mol_module(input_embeddings, item_embeddings, user_ids=user_ids)

    def get_query_component_embeddings(self, input_embeddings, user_ids):
        return self.mol_module.get_query_component_embeddings(input_embeddings, user_ids=user_ids)

    def get_item_component_embeddings(self, item_embeddings):
        return self.mol_module.get_item_component_embeddings(item_embeddings)
