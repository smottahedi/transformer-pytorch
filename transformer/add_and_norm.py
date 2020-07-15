"""Add and Norm"""

import torch
from torch import nn


class AddNorm(nn.Module):
    """Add and Norm layer"""

    def __init__(self, hidden_dem, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dem)

    def forward(self, input_a, input_b):
        return self.ln(self.dropout(input_b) + input_a)
