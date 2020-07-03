""""Multi-head Attention implementation"""

import torch.nn as nn


class MultiheadAttention(nn.Module):
    """Multi-head Attention."""

    def __init__(self, embed_dim, num_hiddens, num_heads, # pylint: disable=too-many-arguments
                 attention, use_bias=False, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.attention = attention
        self.layer_q = nn.Linear(embed_dim, num_hiddens, bias=use_bias)
        self.layer_k = nn.Linear(embed_dim, num_hiddens, bias=use_bias)
        self.layer_v = nn.Linear(embed_dim, num_hiddens, bias=use_bias)
        self.layer_o = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)


    def forward(self, query, key, value, valid_len): # pylint: disable=arguments-differ
        # input shape: (batch_size, seq_len, dim)

        # (batch_size, seq_len, num_hiddens) ->
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        query = transpose_qkv(self.layer_q(query), self.num_heads)
        key = transpose_qkv(self.layer_k(key), self.num_heads)
        value = transpose_qkv(self.layer_v(value), self.num_heads)

        if valid_len is not None:
            if valid_len.ndim == 1:
                valid_len = valid_len.repeat(self.num_heads)
            else:
                valid_len = valid_len(self.num_heads, 1)

        output = self.attention(query, key, value, valid_len)
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        # to (batch_size, seq_len, num_hiddens
        output_concat = transpose_output(output, self.num_heads)
        return self.layer_o(output_concat)


def transpose_qkv(inputs, num_heads):
    """transpose tensor"""

    # (batch_size, seq_len, num_hiddens) ->
    # (batch_size * num_heads, seq_len, num_heads, num_hiddens / num_heads)

    inputs = inputs.reshape([inputs.shape[0], inputs.shape[1], num_heads, -1]).\
            transpose(1, 2)
    output = inputs.reshape([-1, inputs.shape[2], inputs.shape[3]])
    return output


def transpose_output(inputs, num_heads):
    """Reverse transpose_qkv"""

    inputs = inputs.reshape([-1, num_heads, inputs.shape[1], inputs.shape[2]])
    inputs = inputs.transpose(1, 2)
    return inputs.reshape([inputs.shape[0], inputs.shape[1], -1])
