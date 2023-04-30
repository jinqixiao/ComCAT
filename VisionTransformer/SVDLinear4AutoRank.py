# -*- coding:utf-8 -*-
# 
# Author: Jinqi Xiao
# Time: 2022/9/13

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules import Module
from SVDLinear import SVDLinear


class SVDLinear4AutoRank(Module):
    Rank_Step_Size = {192: [32, 64, 96, 128, 160],
                      384: [64, 128, 192, 256, 320],
                      768: [128, 192, 256, 320, 384, 448, 512]}
    # Rank_Step_Size = {192: [6, 12, 24, 48, 64, 96, 128, 144],
    #                   384: [12, 24, 32, 48, 64, 96, 128, 144, 192, 288],
    #                   768: [24, 32, 48, 64, 128, 192, 256, 320, 384, 448, 512]}
    temperature = 5

    def get_choices(self):
        return len(SVDLinear4AutoRank.Rank_Step_Size[min(self.in_features, self.out_features)])

    def __init__(self, fc, rank=None) -> None:
        super().__init__()

        self.in_features = fc.in_features
        self.out_features = fc.out_features

        # self.rank_alpha = Parameter(torch.Tensor([1.0 / self.get_choices() for i in range(self.get_choices())]))
        self.rank_alpha = None
        self.candidate_rank_weights = None
        if rank is None:
            rank = max(SVDLinear4AutoRank.Rank_Step_Size[min(self.in_features, self.out_features)])
        self.selected_rank_layer = SVDLinear(self.in_features, self.out_features, fc.bias is not None,
                                             dense_w=fc.weight.data,
                                             dense_b=fc.bias.data, rank=rank)

    def forward(self, x: Tensor) -> Tensor:
        return self.selected_rank_layer(x)

    @torch.no_grad()
    def decompose_w_by_rank(self):
        ori_matrix = self.selected_rank_layer.last_factor.weight.data @ self.selected_rank_layer.first_factor.weight.data
        u, s, v = np.linalg.svd(ori_matrix.T.detach().cpu().numpy().astype(np.float32), full_matrices=False)
        ranks = SVDLinear4AutoRank.Rank_Step_Size[min(self.in_features, self.out_features)]
        self.candidate_rank_weights = []
        for rank in ranks:
            u1 = u[:, :rank]
            s1 = s[:rank]
            v1 = v[:rank, :]
            self.candidate_rank_weights.append(torch.from_numpy(u1 @ np.diag(s1) @ v1))

    def search_rank_forward(self, x: Tensor, flops_to_accumulate):
        # restore the selected_rank_layer to original matrix
        soft_mask_variables = F.gumbel_softmax(self.rank_alpha, SVDLinear4AutoRank.temperature).to(x.device)
        if self.candidate_rank_weights is None:
            self.decompose_w_by_rank()
        output = sum(m * (x @ w.to(x.device)) for m, w in zip(soft_mask_variables, self.candidate_rank_weights))
        if self.selected_rank_layer.last_factor.bias is not None:
            output += self.selected_rank_layer.last_factor.bias
        ranks = SVDLinear4AutoRank.Rank_Step_Size[min(self.in_features, self.out_features)]
        flops_to_accumulate += sum(
            m * ((self.in_features + self.out_features) * r) for m, r in zip(soft_mask_variables, ranks))
        return output, flops_to_accumulate

    @torch.no_grad()
    def set_rank(self, ranks, final_epoch=False):
        if not final_epoch:
            hard_mask_variables = F.gumbel_softmax(logits=self.rank_alpha, tau=SVDLinear4AutoRank.temperature,
                                                   hard=True).T
            index = int(torch.nonzero(hard_mask_variables))
        else:
            index = int(self.rank_alpha.argmax())
        rank = SVDLinear4AutoRank.Rank_Step_Size[min(self.in_features, self.out_features)][index]
        self.selected_rank_layer = SVDLinear(self.in_features, self.out_features,
                                             self.selected_rank_layer.last_factor.bias is not None,
                                             dense_w=self.candidate_rank_weights[index].T,
                                             dense_b=self.selected_rank_layer.last_factor.bias, rank=rank)
        # print("FC rank: ", rank)
        ranks.append(rank)
        self.candidate_rank_weights = None

    def extra_repr(self):
        s = 'first_FC(in={}, out={}), ' \
            'last_FC(in={}, out={})' \
            .format(self.in_features, self.selected_rank_layer.rank,
                    self.selected_rank_layer.rank, self.out_features)
        return s


if __name__ == '__main__':
    # last = torch.randn([384, 64])
    # first = torch.randn([64, 192])
    # ori_matrix = last @ first
    # u, s, v = np.linalg.svd(ori_matrix.T.detach().cpu().numpy(), full_matrices=False)
    # rank = 64
    # u1 = u[:, :rank]
    # s1 = s[:rank]
    # v1 = v[:rank, :]
    # a = torch.from_numpy(u1 @ np.diag(s1) @ v1)
    # print(a[0, :10])
    # print(ori_matrix.T[0, :10])
    size = 384
    w = torch.randn([size, size])
    b = torch.zeros(size)
    x = torch.randn([1, size])
    loss = torch.nn.MSELoss()
    l_rank = SVDLinear4AutoRank(in_features=size, out_features=size, dense_w=w, dense_b=b)
    lin = torch.nn.Linear(in_features=size, out_features=size)
    lin.weight.data = w
    lin.bias.data = b
    print(loss(l_rank(x), lin(x)))
    o, f = l_rank.search_rank_forward(x, 0)
    print(f)
    print(loss(lin(x), o))
    print(loss(l_rank(x), o))
