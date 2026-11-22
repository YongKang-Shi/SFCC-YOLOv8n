# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "LSKA",
    "LSPA",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fGhost",
    "SFCA",
    "SFCAC2fGhost",
    "iRMB",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))



class LSKA(nn.Module):
    """Large Separable Kernel Attention (LSKA) module used by TGL-YOLO."""

    def __init__(self, c: int, k_size: int = 11):
        super().__init__()
        self.conv = Conv(c, c, k_size, g=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LSPA(nn.Module):
    """Spatial Pyramid Pooling-Fast enhanced with LSKA."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fGhost(C2f):
    """C2f module that replaces each Bottleneck with a GhostBottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize C2fGhost with the same interface as C2f."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))


class SFCA(nn.Module):
    """Support-aware Frequency-Coupled Attention before a neck C2fGhost block."""

    def __init__(
        self,
        c1,
        nc,
        class_image_counts,
        num_train_images,
        epsilon=1.0,
        tau=10.0,
        rarity_max=2.0,
        gate_scale=1.5,
        bottleneck=16,
    ):
        super().__init__()
        if len(class_image_counts) != nc:
            raise ValueError(f"SFCA expected {nc} support counts, got {len(class_image_counts)}")

        counts = torch.tensor(class_image_counts, dtype=torch.float32)
        frequency = (counts + epsilon) / (float(num_train_images) + nc * epsilon)
        positive_rarity = torch.relu(torch.log((1.0 / nc) / frequency))
        reliability = counts / (counts + tau)
        rarity = torch.clamp(reliability * positive_rarity, 0.0, rarity_max) / rarity_max

        self.nc = int(nc)
        self.gate_scale = float(gate_scale)
        self.register_buffer("rarity", rarity.view(1, nc, 1, 1), persistent=True)
        self.response = nn.Sequential(
            Conv(c1, bottleneck, 1, 1),
            nn.Conv2d(bottleneck, nc, 1, bias=True),
        )
        self.spatial_dw = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        self.spatial_bn = nn.BatchNorm2d(3)
        self.spatial_act = nn.SiLU()
        self.spatial_pw = nn.Conv2d(3, 1, 1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1))
        self.capture_response = False
        self.last_response_logits = None

    def forward(self, x):
        q = self.response(x)
        self.last_response_logits = q if self.capture_response else None
        rare_response = (q.sigmoid() * self.rarity).mean(1, keepdim=True)
        descriptor = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True), rare_response), 1)
        attention = torch.sigmoid(self.spatial_pw(self.spatial_act(self.spatial_bn(self.spatial_dw(descriptor)))))
        centered = attention - attention.mean((2, 3), keepdim=True)
        return x * (1.0 + self.gate_scale * self.gamma * centered)


class SFCAC2fGhost(nn.Module):
    """One graph layer implementing Concat -> SFCA -> C2fGhost without changing YOLO layer indices."""

    def __init__(
        self,
        c1,
        c2,
        n,
        nc,
        class_image_counts,
        num_train_images,
        epsilon=1.0,
        tau=10.0,
        rarity_max=2.0,
        gate_scale=1.5,
        bottleneck=16,
    ):
        super().__init__()
        self.sfca = SFCA(
            c1,
            nc,
            class_image_counts,
            num_train_images,
            epsilon,
            tau,
            rarity_max,
            gate_scale,
            bottleneck,
        )
        self.ghost = C2fGhost(c1, c2, n)

    def forward(self, x):
        return self.ghost(self.sfca(x))


class iRMB(nn.Module):
    """Inverted Residual Mobile Block with window attention and depthwise convolution."""

    def __init__(
        self,
        c1,
        c2,
        exp_ratio=1.0,
        dim_head=64,
        window_size=7,
        dw_ks=3,
        attn_s=True,
        qkv_bias=False,
    ):
        """Initialize the iRMB used in the thesis (expansion ratio 1 and 7x7 windows by default)."""
        super().__init__()
        c_mid = int(c1 * exp_ratio)
        if c1 % dim_head != 0:
            raise ValueError(f"iRMB input channels {c1} must be divisible by dim_head {dim_head}")
        self.has_skip = c1 == c2
        self.attn_s = attn_s
        self.window_size = window_size
        self.num_head = c1 // dim_head
        self.scale = dim_head**-0.5
        self.c_mid = c_mid

        self.norm = nn.BatchNorm2d(c1)
        if attn_s:
            self.qk = nn.Conv2d(c1, 2 * c1, 1, bias=qkv_bias)
            self.v = nn.Sequential(nn.Conv2d(c1, c_mid, 1, bias=qkv_bias), nn.ReLU(inplace=True))
        else:
            self.qk = None
            self.v = nn.Sequential(nn.Conv2d(c1, c_mid, 1, bias=qkv_bias), nn.ReLU(inplace=True))

        pad = dw_ks // 2
        self.conv_local = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, dw_ks, 1, pad, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Conv2d(c_mid, c2, 1, bias=False)

    def _window_attention(self, x):
        b0, c0, h0, w0 = x.shape
        ws = self.window_size if self.window_size > 0 else max(h0, w0)
        pad_r = (ws - w0 % ws) % ws
        pad_b = (ws - h0 % ws) % ws
        x = F.pad(x, (0, pad_r, 0, pad_b))
        hp, wp = h0 + pad_b, w0 + pad_r
        nh, nw = hp // ws, wp // ws

        # B,C,Hp,Wp -> (B*nh*nw),C,ws,ws
        x = x.view(b0, c0, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b0 * nh * nw, c0, ws, ws)
        bw, _, _, _ = x.shape
        tokens = ws * ws

        qk = self.qk(x).reshape(bw, 2, self.num_head, c0 // self.num_head, tokens)
        qk = qk.permute(1, 0, 2, 4, 3).contiguous()
        q, k = qk[0], qk[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        v = self.v(x)
        if self.c_mid % self.num_head != 0:
            raise ValueError("iRMB expanded channels must be divisible by the number of attention heads")
        v = v.reshape(bw, self.num_head, self.c_mid // self.num_head, tokens)
        v = v.permute(0, 1, 3, 2).contiguous()
        x = attn @ v
        x = x.permute(0, 1, 3, 2).reshape(bw, self.c_mid, ws, ws)

        # (B*nh*nw),C,ws,ws -> B,C,Hp,Wp
        x = x.view(b0, nh, nw, self.c_mid, ws, ws).permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(b0, self.c_mid, hp, wp)
        return x[:, :, :h0, :w0].contiguous()

    def forward(self, x):
        """Apply normalized window attention, local DWConv, projection, and residual connection."""
        shortcut = x
        x = self.norm(x)
        x = self._window_attention(x) if self.attn_s else self.v(x)
        x = x + self.conv_local(x)
        x = self.proj(x)
        return shortcut + x if self.has_skip else x


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
