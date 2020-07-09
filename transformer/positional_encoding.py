"""Positional Encoding"""

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Positional Encoder"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos = torch.zeros((1, max_len, num_hiddens)) # pylint: disable=no-member
        index = torch.arange(0, max_len).reshape(-1, 1).true_divide(torch.pow( # pylint: disable=no-member
            1000, torch.arange(0, num_hiddens, 2) / num_hiddens))# pylint: disable=no-member
        self.pos[:, :, 0::2] = torch.sin(index.float()) # pylint: disable=no-member
        self.pos[:, :, 1::2] = torch.cos(index.float()) # pylint: disable=no-member

    def forward(self, inputs): # pylint: disable=arguments-differ
        output = inputs + self.pos[:, :inputs.shape[1], :].to(inputs.device)
        return self.dropout(output)
