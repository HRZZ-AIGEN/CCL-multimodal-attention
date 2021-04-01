from abc import ABC
import torch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

"""Multi-modal attention network"""


class MMAtt(torch.nn.Module, ABC):
    def __init__(
            self,
            ppi_depth=3):
        super(MMAtt, self).__init__()
        self.MAT = MAT(
            dim_in=11,
            model_dim=512,
            dim_out=128,
            depth=8,
            Lg=0.33,  # lambda (g)raph - weight for adjacency matrix
            Ld=0.33,  # lambda (d)istance - weight for distance matrix
            La=1,  # lambda (a)ttention - weight for usual self-attention
            dist_kernel_fn='softmax'  # distance kernel fn - either 'exp' or 'softmax'
        )
        # choose different depth of attention layers in the PPI network
        if ppi_depth == 1:
            self.GAT = GAT1(
                dim_in=8,
                model_dim=64,
                dim_out=128
            )

        if ppi_depth == 2:
            self.GAT = GAT2(
                dim_in=8,
                model_dim=64,
                dim_out=128,
            )

        if ppi_depth == 3:
            self.GAT = GAT(
                dim_in=8,
                model_dim=64,
                dim_out=128,
            )

        self.linear_1 = Linear(256, 128)
        self.linear_2 = Linear(128, 1)

    def forward(self, x, adj_mat, dist_mat, mask, x_ppi, ppi_edge_index, ppi_batch):
        x = self.MAT(x=x,
                     adjacency_mat=adj_mat,
                     distance_mat=dist_mat,
                     mask=mask)
        x_ppi, att_ppi = self.GAT(x_ppi, ppi_edge_index, ppi_batch)
        out = torch.cat((x, x_ppi), dim=1)
        out = F.relu(self.linear_1(out))
        out = self.linear_2(out)
        return out, att_ppi


"""
Molecule attention transformer
    Paper: https://arxiv.org/abs/2002.08264
    Implementation: https://github.com/lucidrains/molecule-attention-transformer
"""

DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim=-1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}


def exists(val):
    return val is not None


def default(val, d):
    return d if not exists(val) else val


class Residual(nn.Module, ABC):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, Lg=0.33, Ld=0.33, La=0.33, dist_kernel_fn='exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # hyperparameters controlling the weighted linear combination from
        # self-attention (La)
        # adjacency graph (Lg)
        # pair-wise distance matrix (Ld)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask=None, adjacency_mat=None, distance_mat=None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h=h, qkv=3).unbind(dim=-2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        assert dist_kernel_fn in DIST_KERNELS, f'distance kernel function needs to be one of {DISTANCE_KERNELS.keys()}'
        dist_kernel_config = DIST_KERNELS[dist_kernel_fn]

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(distance_mat):
                # mask distance to infinity
                # todo - make sure for softmax distance kernel, use -infinity
                dist_mask_value = dist_kernel_config['mask_value_fn'](dots)
                distance_mat.masked_fill_(~mask, dist_mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim=-1)
        # attn = torch.nn.functional.dropout(attn, p=0.2, training=True)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        if exists(distance_mat):
            distance_mat = dist_kernel_config['fn'](distance_mat)
            attn = attn + Ld * distance_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# main class

class MAT(nn.Module, ABC):
    def __init__(
            self,
            *,
            dim_in,
            model_dim,
            dim_out,
            depth,
            heads=8,
            Lg=0.33,
            Ld=0.33,
            La=1,
            dist_kernel_fn='exp'
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList([
                Residual(
                    PreNorm(
                        model_dim,
                        Attention(
                            model_dim, heads=heads, Lg=Lg, Ld=Ld, La=La, dist_kernel_fn=dist_kernel_fn
                        )
                    )
                ),
                Residual(
                    PreNorm(
                        model_dim,
                        FeedForward(
                            model_dim
                        )
                    )
                )
            ])
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)

    def forward(
            self,
            x,
            mask=None,
            adjacency_mat=None,
            distance_mat=None
    ):
        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask=mask,
                adjacency_mat=adjacency_mat,
                distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim=-2)
        x = self.ff_out(x)
        return x


"""
Graph attention convolution:
    Paper: https://arxiv.org/abs/1710.10903
    Implementation: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
"""


class GAT(torch.nn.Module, ABC):
    def __init__(self,
                 dim_in=8,
                 model_dim=64,
                 dim_out=128):
        super(GAT, self).__init__()
        self.gat_conv_1 = GATConv(dim_in, model_dim, heads=4, add_self_loops=False)
        self.lin_1 = Linear(dim_in, model_dim * 4)  # Skip connections

        self.gat_conv_2 = GATConv(model_dim * 4, model_dim, heads=4, add_self_loops=False)
        self.lin_2 = Linear(model_dim * 4, model_dim * 4)

        self.gat_conv_3 = GATConv(model_dim * 4, dim_out, heads=6, add_self_loops=False, concat=False)
        self.lin_3 = Linear(model_dim * 4, dim_out)
        self.batch_norm = BatchNorm1d(dim_out)

    def forward(self, x_ppi, ppi_edge_index, ppi_batch):
        x_ppi = F.relu(self.gat_conv_1(x_ppi, ppi_edge_index) + self.lin_1(x_ppi))
        x_ppi = F.relu(self.gat_conv_2(x_ppi, ppi_edge_index) + self.lin_2(x_ppi))
        x_res_3 = self.lin_3(x_ppi)
        x_ppi, att_ppi = self.gat_conv_3(x_ppi, ppi_edge_index, return_attention_weights=True)
        x_ppi = x_ppi + x_res_3  # add last residual
        x_ppi = self.batch_norm(F.relu(x_ppi))
        x_ppi = global_max_pool(x_ppi, ppi_batch)
        return x_ppi, att_ppi


"""Implement lower-depth attention nets"""


# 1 GAT Conv layer
class GAT1(torch.nn.Module, ABC):
    def __init__(self, dim_in=8, model_dim=64, dim_out=128):
        super(GAT1, self).__init__()
        super().__init__()
        self.gat_conv_1 = GATConv(dim_in, dim_out, heads=4, add_self_loops=False, concat=False)
        self.lin_1 = Linear(dim_in, dim_out)  # skip connections
        self.batch_norm = BatchNorm1d(dim_out)

    def forward(self, x_ppi, ppi_edge_index, ppi_batch):
        x_res_1 = self.lin_1(x_ppi)
        x_ppi, att_ppi = self.gat_conv_1(x_ppi, ppi_edge_index, return_attention_weights=True)
        x_ppi = x_ppi + x_res_1  # add the residual
        x_ppi = self.batch_norm(F.relu(x_ppi))
        x_ppi = global_max_pool(x_ppi, ppi_batch)
        return x_ppi, att_ppi


# 2 GAT Conv layers
class GAT2(torch.nn.Module, ABC):
    def __init__(self, dim_in=8, model_dim=64, dim_out=128):
        super(GAT2, self).__init__()
        super().__init__()
        self.gat_conv_1 = GATConv(dim_in, model_dim, heads=4, add_self_loops=False)
        self.lin_1 = Linear(dim_in, model_dim * 4)  # Skip connections

        self.gat_conv_2 = GATConv(model_dim * 4, dim_out, heads=4, add_self_loops=False, concat=False)
        self.lin_2 = Linear(model_dim * 4, dim_out)
        self.batch_norm = BatchNorm1d(dim_out)

    def forward(self, x_ppi, ppi_edge_index, ppi_batch):
        x_ppi = F.relu(self.gat_conv_1(x_ppi, ppi_edge_index) + self.lin_1(x_ppi))
        x_res_2 = self.lin_2(x_ppi)
        x_ppi, att_ppi = self.gat_conv_2(x_ppi, ppi_edge_index, return_attention_weights=True)
        x_ppi = x_ppi + x_res_2  # add the residual
        x_ppi = self.batch_norm(F.relu(x_ppi))
        x_ppi = global_max_pool(x_ppi, ppi_batch)
        return x_ppi, att_ppi
