# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import numpy as np
import torch
from diffusers.models.attention import xformers
from torch import nn


def SVDByRank(dense_w, rank):
    u, s, v = np.linalg.svd(
        dense_w.detach().squeeze().cpu().numpy(), full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    v = v[:rank, :]
    left_factor = torch.from_numpy(u)
    v = np.diag(s) @ v
    right_factor = torch.from_numpy(v)
    return left_factor, right_factor


class CrossAttentionSVD(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """
    alpha = 1

    def __init__(
            self,
            attn,
            rank,
    ):
        super().__init__()
        self.attn = attn
        self.attn.requires_grad_(False)
        self.dim_head = attn.to_q.out_features // attn.heads
        self.upcast_attention = attn.upcast_attention
        self.scale = self.dim_head ** -0.5
        self.heads = attn.heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = attn.sliceable_head_dim
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False

        self.to_q = nn.Linear(attn.to_q.in_features, rank * self.heads, bias=attn.to_q.bias is not None)
        self.to_k = nn.Linear(attn.to_k.in_features, rank * self.heads, bias=attn.to_k.bias is not None)
        self.to_v = nn.Linear(attn.to_v.in_features, rank * self.heads, bias=attn.to_v.bias is not None)
        self.to_out = nn.Linear(rank * self.heads, attn.to_out[0].out_features, bias=False)

    def forward(self, hidden_states, context=None):
        query = self.to_q(hidden_states) * CrossAttentionSVD.alpha
        context = context if context is not None else hidden_states
        key = self.to_k(context) * CrossAttentionSVD.alpha
        value = self.to_v(context) * CrossAttentionSVD.alpha

        B1, N1, C1 = hidden_states.shape
        query = query.view(B1, N1, self.heads, -1).transpose(1, 2)
        B2, N2, C2 = context.shape
        key = key.view(B2, N2, self.heads, -1).transpose(1, 2)
        value = value.view(B2, N2, self.heads, -1).transpose(1, 2)

        attn_query = self.attn.to_q(hidden_states).view(B1, N1, self.heads, -1).transpose(1, 2)
        attn_key = self.attn.to_k(context).view(B2, N2, self.heads, -1).transpose(1, 2)
        attn_value = self.attn.to_v(context).view(B2, N2, self.heads, -1).transpose(1, 2)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when use

        attn = (query @ key.transpose(-2, -1) + attn_query @ attn_key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (attn @ value).transpose(1, 2).reshape(B1, N1, -1)
        x1 = self.to_out(x1) * CrossAttentionSVD.alpha

        x2 = (attn @ attn_value).transpose(1, 2).reshape(B1, N1, -1)
        x2 = self.attn.to_out[0](x2)
        return x1 + x2


class CLIPAttentionSVD(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attn, rank):
        super().__init__()
        self.attn = attn
        self.config = attn.config
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = self.config.attention_dropout
        self.q_proj = nn.Linear(attn.q_proj.in_features, rank * self.num_heads, bias=False)
        self.k_proj = nn.Linear(attn.k_proj.in_features, rank * self.num_heads, bias=False)
        self.v_proj = nn.Linear(attn.v_proj.in_features, rank * self.num_heads, bias=False)
        self.out_proj = nn.Linear(rank * self.num_heads, attn.out_proj.out_features,
                                  bias=attn.out_proj.bias is not None)
        attn.q_proj.bias.data.fill_(0.0)
        attn.k_proj.bias.data.fill_(0.0)
        attn.v_proj.bias.data.fill_(0.0)
        attn.out_proj.bias.data.fill_(0.0)

        self.out_proj.bias.data.fill_(0.0)
        self.qk_bias = nn.Parameter(torch.Tensor([0.0] * self.config.max_position_embeddings))

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal_attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        B, N, C = hidden_states.shape
        query = query.view(B, N, self.num_heads, -1).transpose(1, 2)
        key = key.view(B, N, self.num_heads, -1).transpose(1, 2)
        value = value.view(B, N, self.num_heads, -1).transpose(1, 2)

        attn_query = self.attn.q_proj(hidden_states).view(B, N, self.num_heads, -1).transpose(1, 2)
        attn_key = self.attn.k_proj(hidden_states).view(B, N, self.num_heads, -1).transpose(1, 2)
        attn_value = self.attn.v_proj(hidden_states).view(B, N, self.num_heads, -1).transpose(1, 2)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when use

        attn_weights = (query @ key.transpose(-2, -1) + attn_query @ attn_key.transpose(-2, -1))
        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            attn_weights = attn_weights.view(B, self.num_heads, N, -1) + causal_attention_mask
            # attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        if attention_mask is not None:
            attn_weights = attn_weights.view(B, self.num_heads, N, -1) + attention_mask
            # attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(B, self.num_heads, N, -1)
            # attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, -1)
        else:
            attn_weights_reshaped = None

        attn = (attn_weights + self.qk_bias) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (attn @ value).transpose(1, 2).reshape(B, N, -1)
        x1 = self.out_proj(x1)

        x2 = (attn @ attn_value).transpose(1, 2).reshape(B, N, -1)
        x2 = self.attn.out_proj(x2)
        return x1 + x2, attn_weights_reshaped
