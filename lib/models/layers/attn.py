import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        split_attn = False
        len_t = 49
        if split_attn:
            attn_t = attn[..., :len_t].softmax(dim=-1)
            attn_s = attn[..., len_t:].softmax(dim=-1)
            attn = torch.cat([attn_t, attn_s], dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention for (B, N, C) input without multi-head attention"

    def __init__(self, dim, qk_scale=None, attn_drop=0., kernel_size=3, dilation=1):
        super().__init__()
        self.dim = dim
        self.scale = qk_scale or dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation,
                                padding=dilation * (kernel_size - 1) // 2, stride=1)
        self.attn_drop = nn.Dropout(attn_drop)

        # Projections for q, k, v (here we assume they are the same, i.e. self-attention)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        # B, N, C
        B, N, C = x.shape

        # Project x to q, k, v (since it's self-attention, qkv are projections of x)
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, N, C

        # Unfold k and v for attention
        k = self.unfold(k).reshape(B, N, -1)  # B, N, k*k*C (where k is kernel_size)
        v = self.unfold(v).reshape(B, N, -1, C)  # B, N, k*k, C

        # Flatten k to match v's last two dimensions for matrix multiplication
        k = k.unsqueeze(-2)  # B, N, 1, k*k*C

        # Attention
        attn = (q[:, :, None, :] @ k) * self.scale  # B, N, 1, k*k*C
        attn = attn.softmax(dim=-1)  # Apply softmax over the unfolded dimensions
        attn = attn[:, :, 0, :]  # Remove the singleton dimension
        attn = self.attn_drop(attn)  # Apply dropout

        # Aggregate values
        out = attn[:, :, None, :] @ v  # B, N, 1, k*k*C
        out = out.reshape(B, N, -1)  # B, N, C (since unfolding increased dimensions)

        # Note: This implementation might not perfectly restore the original C dimension
        # depending on the kernel_size and dilation. For perfect restoration, consider
        # adjusting the padding or using a ConvTranspose to upsample the output.

        # However, for simplicity and assuming a reasonable kernel_size and dilation,
        # the above reshape should work well enough for most cases.

        return out


class MultiDilateLocalAttention(nn.Module):
    "Implementation of Multi-Dilation Local Attention"

    def __init__(self, input_size, dim, num_dilations=3, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_dilations = num_dilations
        self.dilation = dilation
        self.scale = qk_scale or (dim ** -0.5)

        # 确保qkv层的输出维度是dim * 3
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # 假设DilateAttention类已经定义，并且可以接受[B, C, N]形状的q, k, v
        self.dilate_attention = nn.ModuleList([
            # 注意：这里可能需要调整DilateAttention的构造函数参数
            DilateAttention(dim, qk_scale=qk_scale, attn_drop=attn_drop, kernel_size=kernel_size, dilation=d)
            for d in dilation
        ])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 检查输入x的形状
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected input dimension C to be {self.dim}, but got {C}")

            # 投影qkv
        qkv = self.qkv(x.view(B, N * C)).reshape(B, 3, C, N).permute(2, 0, 1, 3)  # B, 3*C, N -> B, C, 3, N
        q, k, v = qkv.chunk(3, dim=1)  # 注意这里分割的维度应该是1，因为我们在permute后调整了维度

        # 对每个膨胀率应用DilateAttention
        # 注意：这里假设DilateAttention能够处理[B, C, N]形状的输入
        outputs = []
        for attn in self.dilate_attention:
            attn_out = attn(q.squeeze(1), k.squeeze(1), v.squeeze(1))  # 移除多余的维度
            outputs.append(attn_out)

            # 合并输出
        combined_output = torch.stack(outputs, dim=1).mean(dim=1)  # 平均合并

        # 投影并应用dropout
        x = self.proj(combined_output.reshape(B, N, -1))  # 确保输出形状正确
        x = self.proj_drop(x)

        return x