import torch
from torch import nn


class FlowModel2D(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 2, output_dim: int = 2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.t_emb = nn.Linear(1, hidden_dim)
        self.act1 = torch.nn.GELU()
        self.middle_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = torch.nn.GELU()
        self.middle_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = torch.nn.GELU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        t_emb = self.t_emb(t)

        x = self.input_layer(x) + t_emb
        x = self.act1(x)

        x = self.middle_layer1(x) + t_emb
        x = self.act2(x)

        x = self.middle_layer2(x) + t_emb
        x = self.act3(x)

        x = self.output_layer(x)

        return x
