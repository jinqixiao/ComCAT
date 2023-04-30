# -*- coding:utf-8 -*-
#
# Author: Jinqi Xiao
# Time: 2022/11/27
import numpy as np
import torch
import torch.nn.functional as F


# from vision_transformer import Attention2


class AttentionNewSVD(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, atten, qk_rank, vp_rank, attn2_with_bias=False, attn_drop=0, drop=0):
        super().__init__()
        self.num_heads = atten.num_heads
        self.head_dim = atten.head_dim
        self.scale = self.head_dim ** -0.5
        self.attn2_with_bias = attn2_with_bias
        self.attn_drop_rate = attn_drop
        self.drop_rate = drop

        wq = atten.w_qs.weight.data.T.reshape(atten.w_qs.in_features, self.num_heads, self.head_dim).transpose(0, 1)
        wk = atten.w_ks.weight.data.T.reshape(atten.w_qs.in_features,
                                              self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        w_qk = wq @ wk

        wq_l = []
        wk_r = []
        for i in range(self.num_heads):
            l, r = self.SVDByRank(w_qk[i, :, :], qk_rank)
            wq_l.append(l)
            wk_r.append(r)

        wq = torch.cat(wq_l, 1)
        wk = torch.cat(wk_r, 0).T
        self.w_qs = torch.nn.Linear(atten.dim, wq.shape[1], bias=attn2_with_bias)
        self.w_ks = torch.nn.Linear(atten.dim, wk.shape[1], bias=attn2_with_bias)
        self.w_qs.weight.data = wq.T
        self.w_ks.weight.data = wk.T

        wvi = atten.w_vs.weight.data.T.reshape(atten.w_vs.in_features, self.num_heads,
                                               self.head_dim).transpose(0, 1)
        wi = atten.proj.weight.data.reshape(atten.w_qs.in_features, self.num_heads,
                                            self.head_dim).transpose(0, 1).transpose(1, 2)
        ww = wvi @ wi

        wv_l = []
        w_r = []
        for i in range(self.num_heads):
            l, r = self.SVDByRank(ww[i, :, :], vp_rank)
            wv_l.append(l)
            w_r.append(r)

        wv = torch.cat(wv_l, 1)
        wr = torch.cat(w_r, 0)

        self.w_vs = torch.nn.Linear(atten.dim, wv.shape[1], bias=attn2_with_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(wr.shape[0], atten.dim, bias=attn2_with_bias)
        self.proj_drop = torch.nn.Dropout(drop)

        self.w_vs.weight.data = wv.T
        self.proj.weight.data = wr.T

        if self.attn2_with_bias:
            self.w_qs.bias.data.fill_(0.0)
            self.w_ks.bias.data.fill_(0.0)
            self.w_vs.bias.data.fill_(0.0)
            self.proj.bias.data.fill_(0.0)

    def params(self):
        return self.w_qs.weight.numel() + self.w_ks.weight.numel() + self.w_vs.weight.numel() + self.proj.weight.numel()

    @torch.no_grad()
    def change_rank(self, w_qk, qk_rank, w_vp, vp_rank):
        wq_l = []
        wk_r = []
        qk_rank = qk_rank
        vp_rank = vp_rank
        for i in range(self.num_heads):
            l, r = self.SVDByRank(w_qk[i, :, :], qk_rank)
            wq_l.append(l)
            wk_r.append(r)

        wq = torch.cat(wq_l, 1)
        wk = torch.cat(wk_r, 0).T
        self.w_qs = torch.nn.Linear(self.w_qs.in_features, wq.shape[1], bias=False).to(self.w_qs.weight.device)
        self.w_ks = torch.nn.Linear(self.w_ks.in_features, wk.shape[1], bias=False).to(self.w_ks.weight.device)
        self.w_qs.weight.data = wq.T
        self.w_ks.weight.data = wk.T

        wv_l = []
        w_r = []
        for i in range(self.num_heads):
            l, r = self.SVDByRank(w_vp[i, :, :], vp_rank)
            wv_l.append(l)
            w_r.append(r)

        wv = torch.cat(wv_l, 1)
        wr = torch.cat(w_r, 0)

        self.w_vs = torch.nn.Linear(self.w_vs.in_features, wv.shape[1], bias=False).to(self.w_vs.weight.device)
        self.attn_drop = torch.nn.Dropout(self.attn_drop_rate)
        self.proj = torch.nn.Linear(wr.shape[0], self.w_vs.in_features, bias=False).to(self.proj.weight.device)
        self.proj_drop = torch.nn.Dropout(self.drop_rate)

        self.w_vs.weight.data = wv.T
        self.proj.weight.data = wr.T

    @staticmethod
    def SVDByRank(dense_w, rank):
        u, s, v = np.linalg.svd(
            dense_w.detach().squeeze().cpu().numpy(), full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        v = v[:rank, :]
        left_factor = torch.from_numpy(u)
        v = np.diag(s) @ v
        right_factor = torch.from_numpy(v)
        return left_factor, right_factor

    @staticmethod
    @torch.no_grad()
    def SVD_by_rank(usv, rank):
        u = usv[0][:, :rank]
        s = usv[1][:rank]
        v = usv[2][:rank, :]
        return u @ np.diag(s) @ v

    def forward(self, x):
        B, N, C = x.shape
        # print(self.w_qs.weight.data.shape)
        q = self.w_qs(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.w_ks(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.w_vs(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        # return (q @ k.transpose(-2, -1))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, -1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionNewSVD4AutoRank(torch.nn.Module):
    Rank_Step_Size = {192: [32, 64, 96, 128, 160],
                      384: [64, 128, 192, 256, 320],
                      768: [128, 192, 256, 320, 384, 448, 512]}
    # Rank_Step_Size = {192: [3, 9, 18, 36, 54, 72, 108, 144],
    #                   384: [6, 18, 36, 72, 108, 144, 216, 288],
    #                   768: [12, 36, 72, 144, 216, 288, 432, 576]}
    temperature = 5

    def get_choices(self):
        return len(AttentionNewSVD4AutoRank.Rank_Step_Size[self.in_features])

    def __init__(self, atten, attn2_with_bias=False, qk_rank=None, vp_rank=None) -> None:
        super().__init__()
        self.in_features = atten.w_qs.in_features
        self.out_features = atten.w_qs.out_features
        # self.qk_rank_alpha = torch.nn.Parameter(
        #     torch.Tensor([1.0 / self.get_choices() for i in range(self.get_choices())]))
        # self.vp_rank_alpha = torch.nn.Parameter(
        #     torch.Tensor([1.0 / self.get_choices() for i in range(self.get_choices())]))
        self.qk_rank_alpha = None
        self.vp_rank_alpha = None
        self.candidate_qk_rank_weights = None
        self.candidate_vp_rank_weights = None
        if qk_rank is None or vp_rank is None:
            qk_rank = max(
                AttentionNewSVD4AutoRank.Rank_Step_Size[self.in_features]) // atten.num_heads
            vp_rank = qk_rank
        else:
            qk_rank = qk_rank // atten.num_heads
            vp_rank = vp_rank // atten.num_heads
        self.selected_rank_layer = AttentionNewSVD(atten=atten, qk_rank=qk_rank, vp_rank=vp_rank,
                                                   attn2_with_bias=attn2_with_bias)
        self.w_qks = None
        self.w_vps = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selected_rank_layer(x)

    @torch.no_grad()
    def get_weights_by_svd(self, dense_w, ranks):
        dense_w = dense_w.detach().cpu().numpy()
        weights = []
        usv = []
        for i in range(self.selected_rank_layer.num_heads):
            wi = dense_w[i, :, :]
            usv.append((np.linalg.svd(wi.astype(np.float32), full_matrices=False)))
        for rank in ranks:
            rank = rank // self.selected_rank_layer.num_heads
            tmp = []
            for i in range(self.selected_rank_layer.num_heads):
                z = self.selected_rank_layer.SVD_by_rank(usv[i], rank)
                tmp.append(torch.from_numpy(z))
            weights.append(torch.stack(tmp, 0))
        return weights

    @torch.no_grad()
    def get_weights_by_ranks(self, ranks):
        w_qs = self.selected_rank_layer.w_qs.weight.data.T
        w_ks = self.selected_rank_layer.w_ks.weight.data.T
        w_vs = self.selected_rank_layer.w_vs.weight.data.T
        proj = self.selected_rank_layer.proj.weight.data

        dim = self.selected_rank_layer.w_qs.in_features
        num_heads = self.selected_rank_layer.num_heads
        wq = w_qs.view(dim, num_heads, -1).transpose(0, 1)
        wk = w_ks.view(dim, num_heads, -1).transpose(0, 1).transpose(1, 2)
        wv = w_vs.reshape(dim, num_heads, -1).transpose(0, 1)
        wp = proj.reshape(dim, num_heads, -1).transpose(0, 1).transpose(1, 2)
        w_qk = wq @ wk
        w_vp = wv @ wp

        self.w_qks = self.get_weights_by_svd(w_qk, ranks)
        self.w_vps = self.get_weights_by_svd(w_vp, ranks)

    def search_rank_forward(self, x, params_to_accumulate):
        ranks = AttentionNewSVD4AutoRank.Rank_Step_Size[self.in_features]
        # ranks = [512]
        if self.w_qks is None or self.w_vps is None:
            self.get_weights_by_ranks(ranks)
        qk_soft_mask_variables = F.gumbel_softmax(self.qk_rank_alpha, AttentionNewSVD4AutoRank.temperature)
        vp_soft_mask_variables = F.gumbel_softmax(self.vp_rank_alpha, AttentionNewSVD4AutoRank.temperature)
        # qk_soft_mask_variables = [1]
        # vp_soft_mask_variables = [1]
        attn = 0
        for i in range(len(self.w_qks)):
            w_qk = self.w_qks[i]
            w_qk.to(x.device)
            qk = []
            xt = x.transpose(1, 2)
            for n in range(self.selected_rank_layer.num_heads):
                qk.append(x @ w_qk[n, :, :].to(x.device) @ xt)
            qk = torch.stack(qk, 0).transpose(0, 1)
            attn += qk * self.selected_rank_layer.scale * qk_soft_mask_variables[i]
        attn = F.softmax(attn, dim=-1)
        output = 0
        for i in range(len(self.w_vps)):
            w_vp = self.w_vps[i]
            vp = 0
            for n in range(self.selected_rank_layer.num_heads):
                vp += attn[:, n, :, :].to(x.device) @ x @ w_vp[n, :, :].to(x.device)
            output += vp * vp_soft_mask_variables[i]
        params_to_accumulate += sum(
            m * self.selected_rank_layer.w_qs.in_features * r * 2 for m, r in zip(qk_soft_mask_variables, ranks))
        params_to_accumulate += sum(
            m * self.selected_rank_layer.w_vs.in_features * r * 2 for m, r in zip(vp_soft_mask_variables, ranks))
        return output, params_to_accumulate

    @torch.no_grad()
    def set_rank(self, ranks, use_argmax=False):
        if not use_argmax:
            qk_hard_mask_variables = F.gumbel_softmax(logits=self.qk_rank_alpha,
                                                      tau=AttentionNewSVD4AutoRank.temperature,
                                                      hard=True).T
            qk_index = int(torch.nonzero(qk_hard_mask_variables))
            vp_hard_mask_variables = F.gumbel_softmax(logits=self.vp_rank_alpha,
                                                      tau=AttentionNewSVD4AutoRank.temperature,
                                                      hard=True).T
            vp_index = int(torch.nonzero(vp_hard_mask_variables))
        else:
            qk_index = int(self.qk_rank_alpha.argmax())
            vp_index = int(self.vp_rank_alpha.argmax())
        qk_rank = AttentionNewSVD4AutoRank.Rank_Step_Size[self.in_features][
                      qk_index] // self.selected_rank_layer.num_heads
        vp_rank = AttentionNewSVD4AutoRank.Rank_Step_Size[self.in_features][
                      vp_index] // self.selected_rank_layer.num_heads
        self.selected_rank_layer.change_rank(self.w_qks[qk_index], qk_rank, self.w_vps[vp_index], vp_rank)
        # print("qk_rank, vp_rank: ", self.selected_rank_layer.w_qs.out_features,
        #       self.selected_rank_layer.proj.in_features)
        ranks.append(self.selected_rank_layer.w_qs.out_features)
        ranks.append(self.selected_rank_layer.proj.in_features)
        self.w_qks = None
        self.w_vps = None

    def extra_repr(self):
        s = 'w_qs(in={}, out={}), ' \
            'w_ks(in={}, out={}), ' \
            'w_vs(in={}, out={}), ' \
            'proj(in={}, out={}), ' \
            .format(self.selected_rank_layer.w_qs.in_features, self.selected_rank_layer.w_qs.out_features,
                    self.selected_rank_layer.w_ks.in_features, self.selected_rank_layer.w_ks.out_features,
                    self.selected_rank_layer.w_vs.in_features, self.selected_rank_layer.w_vs.out_features,
                    self.selected_rank_layer.proj.in_features, self.selected_rank_layer.proj.out_features, )
        return s


# if __name__ == '__main__':
#     attn2 = Attention2(dim=768, num_heads=12)
#     rank_attn = AttentionNewSVD4AutoRank(attn2, True)
#     x = torch.randn([96, 197, 768])
#     loss = torch.nn.MSELoss()
#     print(loss(attn2(x), rank_attn(x)))
#     o, f = rank_attn.search_rank_forward(x, 0)
#     print(f)
#     print(loss(attn2(x), o))
#     print(loss(rank_attn(x), o))
