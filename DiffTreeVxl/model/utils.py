import math
import torch
import yaml
# import scipy
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat
from inspect import isfunction
from functools import partial
from tqdm import tqdm


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def activation_function():
    return nn.LeakyReLU()

def normalization(dim):
    return nn.BatchNorm3d(dim)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def update_moving_average(ma_model, current_model, ema_updater):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def beta_linear_log_snr(t):
    return -torch.log(torch.expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s: float = 0.008):
    return -torch.log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def position_encoding(d_model, length):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    config = yaml.safe_load(string)
    return config


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim_list: list, # [in, 1, 2, 3]
                 time_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.dim_list = dim_list
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim_list[1]),
            activation_function(),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(dim_list[0], dim_list[1], kernel_size=1),
            normalization(dim_list[1]),
            activation_function(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(dim_list[1], dim_list[2], kernel_size=3, padding=1),
            normalization(dim_list[2]),
            activation_function(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(dim_list[2], dim_list[3], kernel_size=1),
            normalization(dim_list[3]),
            activation_function(),
            nn.Dropout(dropout),
        )
        self.res_conv = nn.Conv3d(dim_list[0], dim_list[3], 1).apply(weights_init_normal) \
            if dim_list[0] != dim_list[3] else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        # print(h.shape)
        # print(self.time_mlp(t).shape)
        # print(self.time_mlp(t)[(...,) + (None,) *3].shape)
        h = h + self.time_mlp(t)[(...,) + (None,) *3]
        h = self.block2(h)
        # print(h.shape)
        if self.dim_list[1] == self.dim_list[2]:
            h = h + self.time_mlp(t)[(...,) + (None,) * 3]
        h = self.block3(h)
        # print(h.shape)
        return h + self.res_conv(x)

# m = ResnetBlock([64, 32, 32, 64])
# x = torch.randn([5, 64, 64, 64, 64])
# t = torch.randn([5, 256])
# m(x, t)

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        # print(x.shape)
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # print(freqs.shape)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # print(fouriered.shape)
        fouriered = torch.cat((x, fouriered), dim=-1)
        # print(fouriered.shape)
        return fouriered

# m = LearnedSinusoidalPosEmb(22)
# x = torch.randn([22])
# m(x)

class MixImgFeature(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 img_dim,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.img_dim = img_dim
        dim = in_dim + img_dim
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=2, padding=1),
            normalization(dim),
            activation_function(),
            nn.Dropout(dropout),
            nn.Conv3d(dim, out_dim, kernel_size=1),
            normalization(out_dim),
            activation_function(),
            nn.Dropout(dropout),
        )
    def forward(self, x, img):
        vxl_size = x.shape[-1]
        img = rearrange(img, 'b h -> b h 1 1 1')\
            .repeat(1,1,vxl_size,vxl_size,vxl_size)
        # print(x.shape, img.shape)
        x = torch.cat((x, img), dim=1)
        x = self.block(x)
        # print(x.shape)
        return x



class MixImgAttention(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 img_dim,
                 voxel_size,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.img_dim = img_dim
        self.voxel_size = voxel_size
        self.down_factor = 2
        self.q = nn.Sequential(
            nn.BatchNorm3d(in_dim),
            activation_function(),
            nn.Conv3d(in_dim, in_dim, 3, 2, 1),
        ).apply(weights_init_normal)
        self.k = nn.Sequential(
            nn.Linear(img_dim, in_dim),
            nn.LayerNorm(in_dim))
        self.v = nn.Sequential(
            nn.Linear(img_dim, in_dim),
            nn.LayerNorm(in_dim))
        self.voxel_pe = position_encoding(in_dim, (self.voxel_size//self.down_factor)**3)
        self.condition_pe = position_encoding(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mix = nn.Sequential(
            nn.Conv3d(in_dim*2, out_dim, 3, 1, 1),
            normalization(out_dim),
            activation_function(),
            nn.Dropout(dropout),
        )

    def forward(self, x, img):
        # print(self.q(x).shape)
        q = self.q(x).reshape(x.shape[0], self.in_dim, -1).transpose(1,2) \
            + self.voxel_pe.to(x.device).unsqueeze(0)
        k = self.k(img).unsqueeze(1) + self.condition_pe.to(x.device).unsqueeze(0)
        v = self.v(img).unsqueeze(1) + self.condition_pe.to(x.device).unsqueeze(0)
        # print(q.shape, k.shape, v.shape)
        attn, _ = self.attn(q, k, v)
        attn = attn.transpose(1, 2).reshape(x.shape[0], self.in_dim, *(self.voxel_size // 2,) * 3)
        # print(attn.shape)
        x = torch.cat([self.q(x),attn], 1)
        return self.mix(x)

# m = MixImgAttention(32, 64, 256, voxel_size=64)
# x = torch.randn((5, 32, 64, 64, 64))
# img = torch.randn((5, 256))
# print(m(x, img).shape)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
