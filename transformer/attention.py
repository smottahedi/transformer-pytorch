"""Dot-product attention."""

import torch
import torch.nn as nn


def masked_softmax(inputs, valid_len):
    """masked-softmax function."""

    if valid_len is None:
        return nn.functional.softmax(inputs, dim=-1)

    shape = inputs.shape
    if valid_len.dim() == 1:
        valid_len = torch.repeat_interleave(valid_len,
                                            repeats=shape[0],
                                            dim=0)
    else:
        valid_len = valid_len.reshape(-1)

    inputs = inputs.reshape(-1, shape[-1])
    for count, row in enumerate(inputs):
        row[int(valid_len[count]):] = -1e6
    return nn.functional.softmax(inputs.reshape(shape), dim=-1)



class DotProductAttention(nn.Module):
    """Dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)


    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_len: either (batch_size, ) or (batch_size, xx)

    def forward(self, query, key, value, valid_len=None): # pylint: disable=arguments-differ
        dim = query.shape[-1]
        score = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(dim)
        attention_weights = self.dropout(masked_softmax(score, valid_len))
        return torch.bmm(attention_weights, value)
