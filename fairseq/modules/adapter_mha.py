"""Adapter MHA modules."""
import math
from collections import OrderedDict

import torch
from torch import nn
from fairseq.modules.multihead_attention import MultiheadAttention


class AdapterMHA(nn.Module):
    def __init__(
        self,
        input_size,
        embed_dim,
        num_heads,
        add_layer_norm_before_mha=True,
        add_layer_norm_after_mha=True,
        residual_after_mha=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.add_layer_norm_before_mha = add_layer_norm_before_mha
        self.add_layer_norm_after_mha = add_layer_norm_after_mha
        self.residual_after_mha = residual_after_mha

        self.linear1 = nn.Linear(self.input_size, self.embed_dim)

        self.layer_norm_before_mha = None
        if self.add_layer_norm_before_mha:
            self.layer_norm_before_mha = nn.LayerNorm(self.embed_dim)

        self.mha = MultiheadAttention(self.embed_dim, self.num_heads, self_attention=True)

        self.add_layer_norm_after_mha = None
        if self.add_layer_norm_after_mha:
            self.layer_norm_after_mha = nn.LayerNorm(self.embed_dim)

        self.linear2 = nn.Linear(self.embed_dim, self.input_size)


    def forward(self, x, key_padding_mask=None, attn_mask=None):

        x = self.linear1(x)
        residual = x
        if self.add_layer_norm_before_mha:
            x = self.layer_norm_before_mha(x)
        
        x = self.mha(x, x, x, key_padding_mask=key_padding_mask,attn_mask=attn_mask)[0]

        if self.residual_after_mha:
            x = residual + x

        if self.add_layer_norm_after_mha:
            x = self.layer_norm_after_mha(x)

        x = self.linear2(x)

        return x