# -*- coding:utf-8 -*-
# 
# Author: Jinqi Xiao
# Time: 2022/9/13

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import init
from torch.nn.modules import Module


class SVDLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 hp_dict: Optional = None, name: str = None, compression_ratio=None,
                 dense_w: Tensor = None, dense_b: Tensor = None, rank=None, flops=0, latency=0) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if compression_ratio is not None:
            rank = int(in_features * out_features / (compression_ratio * (in_features + out_features)))
        elif rank is None and hp_dict is not None:
            rank = hp_dict.rank[name]
        self.rank = rank

        # self.first_factor = Parameter(torch.Tensor(self.rank, self.in_features))
        # self.last_factor = Parameter(torch.Tensor(self.out_features, self.rank))
        self.first_factor = torch.nn.Linear(self.in_features, self.rank, bias=False)
        self.last_factor = torch.nn.Linear(self.rank, self.out_features, bias=bias)

        if dense_b is not None:
            self.last_factor.bias.data = dense_b

        if dense_w is not None:
            u, s, v = np.linalg.svd(dense_w.T.detach().cpu().numpy(), full_matrices=False)
            u = u[:, :self.rank]
            s = s[:self.rank]
            v = v[:self.rank, :]
            self.first_factor.weight.data = torch.from_numpy(u).T
            self.last_factor.weight.data = torch.from_numpy(np.diag(s) @ v).T
            # print((dense_w.data - torch.matmul(self.last_factor.data, self.first_factor.data)).abs().sum())
            # self.core_tensor.data = core_tensor
        else:
            self.reset_parameters()
        self._latency = latency
        self._params = self.first_factor.weight.numel() + self.last_factor.weight.numel()
        if bias:
            self._params += self.last_factor.bias.numel()
            self.last_factor.bias.data = dense_b.data
            # self.last_factor.bias.data.fill_(0.0)
        self._flops = self._params

    def params(self):
        return self._params

    def latency(self):
        return self._latency

    def flops(self):
        return self._flops

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.first_factor.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.last_factor.weight, a=math.sqrt(5))
        weight = self.first_factor.weight.T @ self.last_factor.weight.T
        if self.last_factor.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.last_factor.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = self.first_factor(x)
        out = self.last_factor(out)
        # out = F.linear(x, self.first_factor)
        # out = F.linear(out, self.last_factor, self.bias)
        return out

    def forward_flops(self, x=None):
        compr_params = (self.first_factor.numel() +
                        self.last_factor.numel())
        compr_flops = compr_params
        base_flops = self.in_features * self.out_features
        self._flops = compr_flops
        return base_flops, compr_flops

    def extra_repr(self):
        s = 'first_fc(in={}, out={}), ' \
            'last_fc(in={}, out={})' \
            .format(self.in_features, self.rank,
                    self.rank, self.out_features)
        return s


if __name__ == '__main__':
    w = torch.randn([625, 256])
    b = torch.zeros(625)
    # b = torch.randn(625)
    t = SVDLinear(in_features=256, out_features=625, compression_ratio=4, dense_w=w, dense_b=b)
    x = torch.randn([1, 256])
    lin = torch.nn.Linear(in_features=256, out_features=625)
    lin.weight.data = w
    lin.bias.data = b
    l = torch.nn.MSELoss()
    print(l(t(x), lin(x)))
