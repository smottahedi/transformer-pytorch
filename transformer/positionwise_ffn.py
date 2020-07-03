"""Position-wise feed-forward network"""

import torch
from torch import nn
from torch.nn import functional as F

class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, hidden_dim, ffn_num_hiddens, pw_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(hidden_dim, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, pw_num_outputs)

    def forward(self, inputs):
        out_1 = self.dense1(inputs)
        print(out_1.shape)
        return F.relu(self.dense2(out_1))