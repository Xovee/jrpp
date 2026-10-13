import math

import torch
import torch.nn.functional as F
from torch import nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class TransformerBlock(nn.Module):
    def __init__(
        self, input_size, d_model=64, n_heads=2, is_layer_norm=True, is_FFN=True, attn_dropout=0.1
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.is_FFN = is_FFN
        self.is_layer_norm = is_layer_norm
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_v))
        self.W_o = nn.Parameter(torch.Tensor(self.d_v * n_heads, input_size))

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(attn_dropout)
        self.activation = GELU()
        self.layer_norm = nn.LayerNorm(input_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_q, mean=0, std=math.sqrt(2.0 / (self.input_size + self.d_model)))
        nn.init.normal_(self.W_k, mean=0, std=math.sqrt(2.0 / (self.input_size + self.d_model)))
        nn.init.normal_(self.W_v, mean=0, std=math.sqrt(2.0 / (self.input_size + self.d_model)))
        nn.init.xavier_normal_(self.W_o)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def feed_forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))

    def scaled_dot_product_attention(self, q, k, v, epsilon=1e-6, attention_bias=None):
        scores = torch.einsum("bqd,bkd->bqk", q, k) / (self.d_k**0.5 + epsilon)
        if attention_bias is not None:
            scores = scores + attention_bias
        weights = F.softmax(scores, dim=-1)
        # A single mask per query/head, shared across its neighbor tokens.
        # Inverted dropout preserves the expected attention contribution.
        mask = self.dropout(torch.ones_like(weights[..., :1]))
        weights = weights * mask
        return weights.bmm(v)

    def multi_head_attention(self, q, k, v, attention_bias=None):
        batch_size, q_len, _ = q.size()
        _, k_len, _ = k.size()
        _, v_len, _ = v.size()

        q = q.matmul(self.W_q).view(batch_size, q_len, self.n_heads, self.d_k)
        k = k.matmul(self.W_k).view(batch_size, k_len, self.n_heads, self.d_k)
        v = v.matmul(self.W_v).view(batch_size, v_len, self.n_heads, self.d_v)

        q = q.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, q_len, self.d_k)
        k = k.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, k_len, self.d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, v_len, self.d_v)

        if attention_bias is not None:
            attention_bias = (
                attention_bias.unsqueeze(1)
                .expand(-1, self.n_heads, q_len, k_len)
                .reshape(batch_size * self.n_heads, q_len, k_len)
            )
        attended = self.scaled_dot_product_attention(q, k, v, attention_bias=attention_bias)
        attended = attended.view(batch_size, self.n_heads, q_len, self.d_v)
        attended = (
            attended.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, q_len, self.n_heads * self.d_v)
        )
        return self.dropout(attended.matmul(self.W_o))

    def forward(self, q, k, v, attention_bias=None):
        attended = self.multi_head_attention(q, k, v, attention_bias=attention_bias)
        if self.is_layer_norm:
            output = self.layer_norm(q + attended)
            if self.is_FFN:
                output = self.layer_norm(self.feed_forward(output) + output)
            return output

        output = q + attended
        if self.is_FFN:
            output = self.feed_forward(output) + output
        return output
