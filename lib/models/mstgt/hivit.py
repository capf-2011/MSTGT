""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

Modified by Botao Ye
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple
from lib.models.mstgt.base_backbone import BaseBackbone


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, N
        B, d, N = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, self.kernel_size, N // self.kernel_size]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = k.reshape([B, d // self.head_dim, self.head_dim, self.kernel_size, N // self.kernel_size]).permute(0, 1, 4, 2, 3)  # B,h,N,d,k*k   [3, 4, 64, 3, 128]
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = v.reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size, N // self.kernel_size]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k,d
        x = (attn @ v).transpose(1, 2).reshape(B, N, d)
        return x


class MultiDilateLocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, input_size, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        # self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 检查输入x的形状
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected input dimension C to be {self.dim}, but got {C}")
        x = x.clone()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_dilation, C // self.num_dilation).permute(3, 2, 0, 4, 1)  # num_dilation,3,B,C//num_dilation,N

        x = x.permute(0, 2, 1)  # B, C, N
        # qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, N).permute(2, 1, 0, 3, 4)  # num_dilation,3,B,C//num_dilation,N
        # num_dilation,3,B,C//num_dilation,N

        x = x.reshape(B, self.num_dilation, C // self.num_dilation, N).permute(1, 0, 3, 2)  # num_dilation,B,N,C//num_dilation
        # num_dilation, B, N, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, N,C//num_dilation
        x = x.permute(1, 2, 0, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"

    def __init__(self, input_size, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilateLocalAttention(input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):

        # def _inner_forward(x):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

        # if self.use_checkpoint:
        #     x = checkpoint.checkpoint(_inner_forward, x)
        # else:
        #     x = _inner_forward(x)

        # return x


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., rpe=True, use_checkpoint=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        # print(x.shape)  #torch.Size([3, 64, 4, 4, 128])    ste hou torch.Size([3, 384, 512])
        def _inner_forward(x):
            if self.attn is not None:
                x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
                # print(x.shape)  #torch.Size([3, 384, 512])
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        # print(x.shape)
        if self.use_checkpoint:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        # print(x.shape)  #torch.Size([3, 384, 512])  torch.Size([3, 353, 512])
        return x
 

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape  #torch.Size([3, 3, 128, 128])
        # print(x.shape)
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1, 
            patches_resolution[0], self.inner_patches, 
            patches_resolution[1], self.inner_patches, 
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
    
    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :] 
        x1 = x[..., 1::2, 0::2, :] 
        x2 = x[..., 0::2, 1::2, :] 
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchEmbed1(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        B, C, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = x.transpose(1,2).view(B, C, H, W)
        return x


class HiViT(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, 
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
            relative_coords[:, :, 0] = relative_coords[:, :, 0] + Hp - 1
            relative_coords[:, :, 1] += relative_coords[:, :, 1] + Wp - 1
            relative_coords[:, :, 0] *= relative_coords[:, :, 0] * (2 * Wp - 1)
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))  # stochastic depth decay rule

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for i in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale, 
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),  
                        rpe=rpe, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        
        self.patch_embed14 = PatchEmbed1(
            img_size=img_size, patch_size=patch_size-2, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed16 = PatchEmbed1(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed18 = PatchEmbed1(
            img_size=img_size, patch_size=patch_size+2, in_chans=in_chans, embed_dim=embed_dim)

        self.STE_norm = norm_layer(embed_dim)
        # if nhead == 8:
        #     nhead=nhead+1
        self.STE = DilateBlock(Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,)
        self.MSVC = nn.Linear(embed_dim, embed_dim,bias=False)
        self.MSTG = nn.Linear(embed_dim, embed_dim,bias=False)
        self.norm_ = norm_layer(embed_dim)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

def _create_vision_transformer(pretrained=False, default_cfg=None, **kwargs):
    
    model = HiViT(**kwargs)

    if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            # print(missing_keys, unexpected_keys)
            print('Load pretrained model from: ' + pretrained)

    return model


def hivit_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    
    return model

def hivit_small(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dim=384, depths=[2, 2, 20], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., 
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    
    return model

def hivit_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dim=384, depths=[1, 1, 10], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., 
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    
    return model
