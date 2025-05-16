from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.mstgt.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None
        self.pos_embed_d = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        dynamic_size = to_2tuple(cfg.DATA.DYNAMIC.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        # print(self.absolute_pos_embed.shape)
        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        # for dynamic region
        H, W = dynamic_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        dynamic_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        dynamic_patch_pos_embed = dynamic_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_d = nn.Parameter(dynamic_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)


        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x_list, frame_id, mask=None):
        # print(z.shape) #torch.Size([3, 3, 128, 128])
        avgpool = nn.AvgPool2d(kernel_size=3, stride=3)
        z18 = self.patch_embed18(z)
        z16 = self.patch_embed16(z)
        z14 = self.patch_embed14(z)
        _, _, w, h = z16.shape
        z18 = avgpool(z18).flatten(2).transpose(1, 2)
        z16 = avgpool(z16).flatten(2).transpose(1, 2)
        z14 = avgpool(z14).flatten(2).transpose(1, 2)
        prompt_ms = torch.cat([z18, z16, z14], dim=-2)
        prompt_ms = self.MSPG(prompt_ms)

        z = self.patch_embed(z)
        x = torch.cat(x_list, dim=0)
        x = self.patch_embed(x)
        for blk in self.blocks[:-self.num_main_blocks]:
            z = blk(z)
            x = blk(x)

        z = z[..., 0, 0, :]
        x = x[..., 0, 0, :]

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        
        num = len(x_list)
        _, N, C = x.shape
        B, _, _ = z.shape
        x_list = x.view(num, B, N, C)   #4，3，256，512

        if frame_id<=1: # -1:train or 1:init
            self.STT = z   #torch.Size([3, 64, 512])
        
        xf = []
        while num: # 1： test, >1:train
            x = x_list[-num]
            avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
            # stt = self.STT
            # print(stt.shape)
            prompt_st = avgpool(self.STT.transpose(1, 2).view(B,C,w,h)).flatten(2).transpose(1, 2)
            # print(prompt_st.shape)  #torch.Size([3, 16, 512])
            prompt_st = self.STPG(prompt_st)
            # print(prompt_st.shape)
            prompt = combine_tokens(prompt_ms, prompt_st, mode=self.cat_mode)
            prompt = combine_tokens(prompt, z, mode=self.cat_mode)
            # print(prompt_st.shape)   #torch.Size([3, 16, 512])
            x_h = combine_tokens(x, self.STT, mode=self.cat_mode)
            # print(x_h.shape)  #torch.Size([3, 320, 512])
            z_x_h = combine_tokens(z, x_h, mode=self.cat_mode)
            # print(z_x_h.shape) #torch.Size([3, 384, 512])

            z_x_h = self.pos_drop(z_x_h)
            z_x_h = self.STE(self.STE_norm(z_x_h))  #torch.Size([3, 384, 512])


            # print(z_x_h.shape)
            self.STT = z_x_h[:, :lens_z]#.detach()
            # sttz = self.STT
            # print(sttz.shape)
            x = combine_tokens(prompt, x, mode=self.cat_mode)
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
                x = blk(x)
            xf.append(x)
            num -= 1  #torch.Size([3, 353, 512])

        x = torch.cat(xf, dim=0)
        # x = recover_tokens(x, lens_x, lens_z, mode=self.cat_mode)
        x = self.norm_(x)
        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z, x, frame_id, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        if not isinstance(x,list):
            x = [x] # test, x:torch.Tensor, train,x:dict
        x, aux_dict = self.forward_features(z, x, frame_id)

        return x, aux_dict
