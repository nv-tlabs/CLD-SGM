# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from . import utils


@utils.register_model(name='mlp')
class MLP(nn.Module):
    def __init__(self,
                 config,
                 input_dim=2,
                 index_dim=1,
                 hidden_dim=128):

        super().__init__()

        act = nn.SiLU()

        self.x_input = True
        self.v_input = True if config.sde == 'cld' else False

        if self.x_input and self.v_input:
            in_dim = input_dim * 2 + index_dim
        else:
            in_dim = input_dim + index_dim
        out_dim = input_dim

        self.main = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, out_dim))

    def forward(self, u, t):
        h = torch.cat([u, t.reshape(-1, 1)], dim=1)
        output = self.main(h)

        return output

@utils.register_model(name='resnet')
class ResNet(nn.Module):
    def __init__(self,
                 config,
                 input_dim=2,
                 index_dim=1,
                 hidden_dim=64,
                 n_hidden_layers=20):

        super().__init__()

        self.act = nn.SiLU()
        self.n_hidden_layers = n_hidden_layers

        self.x_input = True
        self.z_input = True if config.sde == 'cld' else False

        if self.x_input and self.z_input:
            in_dim = input_dim * 2 + index_dim
        else:
            in_dim = input_dim + index_dim
        out_dim = input_dim

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim + index_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim + index_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def _append_time(self, h, t):
        time_embedding = torch.log(t)
        return torch.cat([h, time_embedding.reshape(-1, 1)], dim=1)

    def forward(self, u, t):
        h0 = self.layers[0](self._append_time(u, t))
        h = self.act(h0)

        for i in range(self.n_hidden_layers):
            h_new = self.layers[i + 1](self._append_time(h, t))
            h = self.act(h + h_new)

        return self.layers[-1](self._append_time(h, t))
