"""Decoder implementation"""
import torch
from torch import nn

from transformer.attention import DotProductAttention
from transformer.multihead_attention import MultiheadAttention
from transformer.add_and_norm import AddNorm
from transformer.positional_encoding import PositionalEncoding
from transformer.positionwise_ffn import PositionWiseFFN


class DecoderBlock(nn.Module):
    """Decoder"""

    def __init__(self, embed_dim, num_hiddens,
                 ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention_1 = MultiheadAttention(embed_dim, num_hiddens, num_heads,
                                              DotProductAttention(dropout))
        self.addnorm_1 = AddNorm(num_hiddens, dropout)
        self.attention_2 = MultiheadAttention(embed_dim, num_hiddens, num_heads,
                                              DotProductAttention(dropout))
        self.addnorm_2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm_3 = AddNorm(num_hiddens, dropout)

    def forward(self, inputs, state):
        enc_outputs, enc_valid_len = state[0], state[1]
        if state[2][self.i] is None:
            key_values = inputs

        else:
            key_values = torch.cat([state[2][self.i], inputs], axis=1)

        state[2][self.i] = key_values
        
        if self.training:
            batch_size, seq_len, _ = inputs.shape
            valid_len = torch.arange(1, seq_len + 1).repeat(batch_size, 1).to(inputs.device)

        else:
            valid_len = None 
        
        inputs_2 = self.attention_1(inputs, key_values, key_values, valid_len)
        output = self.addnorm_1(inputs, inputs_2)
        output_1 = self.attention_2(output, enc_outputs, enc_outputs, enc_valid_len)
        output_2 = self.addnorm_2(output, output_1)
        output_3 = self.ffn(output_2)
        output_3 = self.addnorm_3(output_2, output_3)
        
        return output_3, state