"""
NAFNet / Real-ESRGAN 模型定义

整合自 D:\\edge\\projects 两个项目的核心模型架构。

项目源:
  - NAFNet (v1.2.0): megvii-research/NAFNet, ECCV 2022
  - Real-ESRGAN (v0.3.0): xinntao/Real-ESRGAN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

# ============================================================================
# 工具模块
# ============================================================================

class LayerNormFunction(torch.autograd.Function):
    """自定义 LayerNorm 前向/反向 (来自 NAFNet arch_util)."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """2D Layer Normalization (来自 NAFNet)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """初始化网络权重 (Kaiming Normal)."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block: nn.Module, num_basic_block: int, **kwarg) -> nn.Sequential:
    """堆叠相同模块."""
    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """无 BN 的残差块: Conv-ReLU-Conv + 残差连接."""

    def __init__(self, num_feat: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Module):
    """上采样模块 (支持 2^n 和 3 倍)."""

    def __init__(self, scale: int, num_feat: int):
        super().__init__()
        m = []
        if (scale & (scale - 1)) == 0:  # 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'不支持的放大倍数: {scale}. 支持 2^n 和 3.')
        super().__init__(*m)


class Local_Base:
    """局部推理基类 (用于大图分块推理)."""

    def convert(self, base_size, train_size, fast_imp=False):
        self.base_size = base_size
        self.train_size = train_size
        if fast_imp:
            return
        # 预计算 local_weight
        convert_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, LayerNorm2d)):
                convert_dict[name] = module
            elif hasattr(module, 'convert'):
                convert_dict[name] = module
        self.convert_dict = convert_dict


# ============================================================================
# NAFNet 核心模块
# ============================================================================

class SimpleGate(nn.Module):
    """简单门控: 将特征沿通道分成两半, 逐元素相乘.
    NAFNet 的核心创新: 用乘法替代非线性激活函数.
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet 基本块.

    结构:
      LayerNorm → 1x1 Conv → 3x3 DW Conv → SimpleGate → SCA → 1x1 Conv → Drop
      → LayerNorm → 1x1 Conv → SimpleGate → 1x1 Conv → Drop
    
    特点: 无显式非线性激活函数 (ReLU/GELU/Sigmoid), 全部用乘法替代.
    """

    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2,
                 drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * DW_Expand

        # --- 通道混合模块 ---
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )

        self.sg = SimpleGate()

        # --- FFN 模块 ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # 可学习的残差缩放
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # --- 通道混合路径 ---
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # --- FFN 路径 ---
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet: Nonlinear Activation Free Network for Image Restoration.

    U-Net 风格编码器-解码器结构, 中间有多个 NAFBlock.
    
    Args:
        img_channel (int): 输入图像通道数, 默认 3 (RGB).
        width (int): 网络宽度 (初始通道数), 默认 16.
        middle_blk_num (int): 中间层块数, 默认 1.
        enc_blk_nums (list[int]): 各编码器阶段块数.
        dec_blk_nums (list[int]): 各解码器阶段块数.
    """

    def __init__(self, img_channel: int = 3, width: int = 16,
                 middle_blk_num: int = 1,
                 enc_blk_nums: List[int] = [],
                 dec_blk_nums: List[int] = []):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # 编码器: 每阶段 NAFBlock stack → 2x 下采样
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(
                *[NAFBlock(chan) for _ in range(num)]
            ))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        # 中间层
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # 解码器: 2x 上采样 → NAFBlock stack
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(
                *[NAFBlock(chan) for _ in range(num)]
            ))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        """将输入 padding 到 self.padder_size 的整数倍."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        # 编码
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # 中间
        x = self.middle_blks(x)

        # 解码 (跳跃连接)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp  # 全局残差连接
        return x[:, :, :H, :W]


class NAFNetLocal(Local_Base, NAFNet):
    """NAFNet 局部推理版 (支持超大图分块处理)."""

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# ============================================================================
# Baseline 模块 (带传统激活函数的基线模型)
# ============================================================================

class BaselineBlock(nn.Module):
    """Baseline 基本块 (与 NAFBlock 结构相同但使用 GELU + SE 注意力).

    用于对比实验: 证明去掉非线性激活后性能反而不降.
    """

    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel, c, 1, bias=True)

        # SE 通道注意力 (含 ReLU + Sigmoid)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, dw_channel // 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dw_channel // 2, dw_channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class Baseline(nn.Module):
    """Baseline 网络 (传统激活函数版).

    与 NAFNet 结构完全相同, 只是 NAFBlock 换成 BaselineBlock.
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[], dw_expand=1, ffn_expand=2):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
            ))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
            ))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


# ============================================================================
# NAFSSR (立体图像超分辨率)
# ============================================================================

class SCAM(nn.Module):
    """Stereo Cross Attention Module (立体交叉注意力模块).

    用于左右视图之间的特征交互.
    """

    def __init__(self, c: int):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)

        attn = torch.matmul(Q_l, Q_r_T) * self.scale
        F_r2l = torch.matmul(torch.softmax(attn, dim=-1), V_r)
        F_l2r = torch.matmul(torch.softmax(attn.permute(0, 1, 3, 2), dim=-1), V_l)

        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r


class NAFBlockSR(nn.Module):
    """用于超分任务的 NAFBlock (可选立体交叉注意力融合)."""

    def __init__(self, c, fusion=False, drop_out_rate=0.0):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple(self.blk(x) for x in feats)
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class NAFNetSR(nn.Module):
    """NAFNet for Super-Resolution (超分辨率版本).

    支持单图超分和立体 (双目) 超分.
    
    Args:
        up_scale (int): 放大倍数, 默认 4.
        width (int): 网络宽度, 默认 48.
        num_blks (int): 主体块数, 默认 16.
        img_channel (int): 图像通道数, 默认 3.
        fusion_from / fusion_to: 立体融合的范围.
        dual (bool): 是否双输入 (立体).
    """

    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3,
                 drop_path_rate=0.0, drop_out_rate=0.0,
                 fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        # 主体: 多个 NAFBlockSR (带可选 DropPath)
        body = []
        for i in range(num_blks):
            blk = NAFBlockSR(width, fusion=(fusion_from <= i <= fusion_to),
                             drop_out_rate=drop_out_rate)
            if drop_path_rate > 0:
                blk = DropPath(drop_path_rate, blk)
            body.append(blk)
        self.body = nn.Sequential(*body) if not body else MySequential(*body)

        self.up = nn.Sequential(
            nn.Conv2d(width, img_channel * up_scale ** 2, 3, padding=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp,)
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out


class NAFSSR(Local_Base, NAFNetSR):
    """NAFSSR: 立体超分辨率网络 (NTIRE 2022 冠军)."""

    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False,
                 fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from,
                          fusion_to=fusion_to, dual=True, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) 模块."""

    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats
        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.
        if self.training and factor != 1.:
            new_feats = tuple(x + factor * (new_x - x)
                              for x, new_x in zip(feats, new_feats))
        return new_feats


class MySequential(nn.Sequential):
    """支持多输入/输出的 Sequential."""

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# ============================================================================
# Real-ESRGAN 模型
# ============================================================================

class RRDBNet(nn.Module):
    """RRDB (Residual-in-Residual Dense Block) 网络.

    Real-ESRGAN 的 backbone, 基于 ESRGAN 的 RRDB 架构.
    
    Args:
        num_in_ch (int): 输入通道数.
        num_out_ch (int): 输出通道数.
        num_feat (int): 中间特征通道数.
        num_block (int): RRDB 块数.
        num_grow_ch (int): 每个 Dense Block 的增长通道数.
        scale (int): 放大倍数.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat,
                               num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 上采样
        self.upsample = Upsample(scale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # 全局残差
        feat = self.upsample(feat)
        out = self.conv_last(feat)
        return out


class DenseBlock(nn.Module):
    """密集连接块."""

    def __init__(self, num_feat, num_grow_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2  # 残差缩放


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block.

    由 3 个 DenseBlock 级联 + 残差连接.
    """

    def __init__(self, num_feat, num_grow_ch):
        super().__init__()
        self.rdb1 = DenseBlock(num_feat, num_grow_ch)
        self.rdb2 = DenseBlock(num_feat, num_grow_ch)
        self.rdb3 = DenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class SRVGGNetCompact(nn.Module):
    """紧凑型 VGG 风格网络.

    用于 Real-ESRGAN 的动漫视频模型 (realesr-animevideov3) 和
    通用模型 (realesr-general-x4v3).
    
    Args:
        num_in_ch (int): 输入通道数.
        num_out_ch (int): 输出通道数.
        num_feat (int): 特征通道数.
        num_conv (int): 卷积层数.
        upscale (int): 放大倍数.
        act_type (str): 激活函数类型 (prelu / relu).
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.body = nn.ModuleList()
        # 第一层
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # 中间层
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        # 最后一层
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale ** 2, 3, 1, 1))
        self.act = nn.PReLU(num_parameters=1, init=0.2) if act_type == 'prelu' else nn.ReLU(inplace=True)
        self.upsample = nn.PixelShuffle(upscale)

    def forward(self, x):
        feat = x
        for i, layer in enumerate(self.body):
            feat = layer(feat)
            if i < len(self.body) - 1:
                feat = self.act(feat)
        feat = self.upsample(feat)
        return feat


# ============================================================================
# 损失函数
# ============================================================================

class PSNRLoss(nn.Module):
    """PSNR Loss: -10 * log10(MSE), 优化目标直接对应 PSNR 指标.

    可选 toY: 转换到 Y 通道 (亮度) 计算.
    """

    def __init__(self, loss_weight=1.0, toY=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1, keepdim=True) + 16.
            target = (target * self.coef).sum(dim=1, keepdim=True) + 16.
            pred, target = pred / 255., target / 255.

        return self.loss_weight * self.scale * \
            torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 的平滑版本)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps).mean()


# ============================================================================
# 模型工厂
# ============================================================================

def create_nafnet(model_size='small'):
    """创建预置配置的 NAFNet.

    Args:
        model_size: 'small' | 'middle' | 'large' | 'baseline'
    
    Returns:
        NAFNet: 配置好的模型实例
    """
    configs = {
        'small': {
            'width': 16,
            'enc_blk_nums': [1, 1, 1, 28],
            'middle_blk_num': 1,
            'dec_blk_nums': [1, 1, 1, 1],
        },
        'middle': {
            'width': 32,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
        },
        'large': {
            'width': 64,
            'enc_blk_nums': [2, 4, 8, 16],
            'middle_blk_num': 32,
            'dec_blk_nums': [2, 2, 2, 2],
        },
        'baseline': {
            'width': 32,
            'enc_blk_nums': [1, 1, 1, 28],
            'middle_blk_num': 1,
            'dec_blk_nums': [1, 1, 1, 1],
        },
    }
    cfg = configs[model_size]
    return NAFNet(img_channel=3, **cfg)


def create_rrdbnet(model_name='RealESRGAN_x4plus'):
    """创建预置配置的 RRDBNet.

    Args:
        model_name: 'RealESRGAN_x4plus' | 'RealESRGAN_x4plus_anime_6B' |
                    'RealESRGAN_x2plus'
    
    Returns:
        RRDBNet: 配置好的模型实例
    """
    configs = {
        'RealESRGAN_x4plus': {'num_block': 23, 'scale': 4},
        'RealESRNet_x4plus': {'num_block': 23, 'scale': 4},
        'RealESRGAN_x4plus_anime_6B': {'num_block': 6, 'scale': 4},
        'RealESRGAN_x2plus': {'num_block': 23, 'scale': 2},
    }
    cfg = configs[model_name]
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_grow_ch=32, **cfg)
