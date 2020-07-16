"""Encoder implementation."""

import math
import torch
from torch import nn
from transformer.multihead_attention import MultiheadAttention  # pylint: disable=import-error
from transformer.add_and_norm import AddNorm
from transformer.positionwise_ffn import PositionWiseFFN
from transformer.positional_encoding import PositionalEncoding
from transformer.attention import DotProductAttention


class EncoderBlock(nn.Module):
    """Encoder block implementation"""

    def __init__(self, embed_dim, num_hiddens, ffn_num_hiddens,
                 num_heads, attention, dropout, use_bias=False, **kwargs):

        super(EncoderBlock, self).__init__(**kwargs)

        self.attention = MultiheadAttention(embed_dim, num_hiddens, num_heads,
                                            attention, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, inputs, valid_len):
        output = self.addnorm1(inputs,  self.attention(
            inputs, inputs, inputs, valid_len))
        return self.addnorm2(output, self.ffn(output))


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, vocab_size, embed_dim, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        self.blks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blks.append(
                EncoderBlock(embed_dim, num_hiddens, ffn_num_hiddens, num_heads, DotProductAttention(dropout),
                             dropout, use_bias)
            )

    def forward(self, inputs, valid_len, *args):
        inputs = self.pos_encoding(self.embedding(inputs.long()) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            inputs = blk(inputs, valid_len)
        return inputs

