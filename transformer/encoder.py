"""Encoder implementation."""

import torch
from torch import nn
from transformer.multihead_attention import MultiheadAttention # pylint: disable=import-error
from transformer.add_and_norm import AddNorm
from transformer.positionwise_ffn import PositionWiseFFN

class EncoderBlock(nn.Module):
    """Encoder block implementation"""

    def __init__(self, embed_dim, num_hiddens, seq_len,  ffn_num_hiddens,
                num_heads, attention, dropout, use_bias=False, **kwargs):

        super(EncoderBlock, self).__init__(**kwargs)

        self.attention = MultiheadAttention(embed_dim, num_hiddens, num_heads, 
                                            attention, use_bias)
        self.addnorm1 = AddNorm(seq_len, num_hiddens, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(seq_len, num_hiddens, dropout)


    def forward(self, inputs, valid_len):
        output = self.addnorm1(inputs,  self.attention(inputs, inputs, inputs, valid_len))
        return self.addnorm2(output, self.ffn(output))