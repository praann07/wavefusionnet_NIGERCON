"""WaveFusion-Net: dual-branch spatial-wavelet network for blind image deblurring.

This file contains the model exactly as it was trained in the notebooks under
``train/`` (the three full-model notebooks share an identical architecture).

Notes on the implementation
---------------------------
* The spatial branch is a U-shaped encoder-decoder of NAFBlocks with a StripAttention
  bottleneck. The wavelet branch applies a fixed Haar DWT and fuses its LL features with
  the spatial features at two encoder scales through ``CrossBranchFusion``.
* ``WaveletBlock`` also computes a high-frequency path, and the branch computes a third
  wavelet level. Their outputs are not used by ``forward``, so they do not influence the
  result (see the Limitations section of the paper). They are kept so that the released
  checkpoints load with ``strict=True``.
* The input is reflection-padded to a multiple of 8 and the output is cropped back.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== WAVELET TRANSFORMS ========================
class DWT(nn.Module):
    """Discrete Wavelet Transform using Haar wavelets"""

    def __init__(self):
        super().__init__()
        self.register_buffer("ll", torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2))
        self.register_buffer("lh", torch.tensor([[0.5, 0.5], [-0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer("hl", torch.tensor([[0.5, -0.5], [0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer("hh", torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))

        x = x.contiguous().view(B * C, 1, x.shape[2], x.shape[3])
        ll = F.conv2d(x, self.ll, stride=2)
        lh = F.conv2d(x, self.lh, stride=2)
        hl = F.conv2d(x, self.hl, stride=2)
        hh = F.conv2d(x, self.hh, stride=2)

        ll = ll.view(B, C, ll.shape[2], ll.shape[3])
        lh = lh.view(B, C, lh.shape[2], lh.shape[3])
        hl = hl.view(B, C, hl.shape[2], hl.shape[3])
        hh = hh.view(B, C, hh.shape[2], hh.shape[3])
        return ll, lh, hl, hh


class IDWT(nn.Module):
    """Inverse Discrete Wavelet Transform"""

    def __init__(self):
        super().__init__()
        self.register_buffer("ll", torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2))
        self.register_buffer("lh", torch.tensor([[0.5, 0.5], [-0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer("hl", torch.tensor([[0.5, -0.5], [0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer("hh", torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2))

    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        ll = ll.view(B * C, 1, H, W)
        lh = lh.view(B * C, 1, H, W)
        hl = hl.view(B * C, 1, H, W)
        hh = hh.view(B * C, 1, H, W)

        ll = F.conv_transpose2d(ll, self.ll, stride=2)
        lh = F.conv_transpose2d(lh, self.lh, stride=2)
        hl = F.conv_transpose2d(hl, self.hl, stride=2)
        hh = F.conv_transpose2d(hh, self.hh, stride=2)

        out = (ll + lh + hl + hh).view(B, C, ll.shape[2], ll.shape[3])
        return out


# ======================== ATTENTION MODULES ========================
class SimpleGate(nn.Module):
    """Simple gating mechanism - splits channels and applies element-wise product"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class StripAttention(nn.Module):
    """Strip attention - processes horizontal and vertical strips separately"""

    def __init__(self, channels, strip_size=7):
        super().__init__()
        self.h_conv = nn.Conv2d(channels, channels, (1, strip_size), padding=(0, strip_size // 2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (strip_size, 1), padding=(strip_size // 2, 0), groups=channels)
        self.proj = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_out = self.h_conv(x)[:, :, :H, :W]
        v_out = self.v_conv(x)[:, :, :H, :W]
        h_attn = torch.sigmoid(h_out)
        v_attn = torch.sigmoid(v_out)
        combined = torch.cat([x * h_attn, x * v_attn], dim=1)
        return self.proj(combined)


class SCA(nn.Module):
    """Simplified Channel Attention"""

    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.fc(self.gap(x))


# ======================== CORE BLOCKS ========================
class NAFBlock(nn.Module):
    """NAFNet-style block with simplified attention"""

    def __init__(self, channels, dw_expand=2):
        super().__init__()
        dw_channels = channels * dw_expand

        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)

        self.sca = SCA(dw_channels // 2)
        self.sg = SimpleGate()

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)

        return x + residual


class WaveletBlock(nn.Module):
    """Process wavelet coefficients with attention to high-frequency details"""

    def __init__(self, channels):
        super().__init__()
        self.ll_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.hf_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=3),
            nn.GELU(),
            nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=3),
        )
        self.hf_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels * 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, ll, lh, hl, hh):
        ll_out = ll + self.ll_conv(ll)

        hf = torch.cat([lh, hl, hh], dim=1)
        hf_feat = self.hf_conv(hf)
        hf_attn = self.hf_attn(hf_feat)
        hf_out = hf + hf_feat * hf_attn

        lh_out, hl_out, hh_out = hf_out.chunk(3, dim=1)
        return ll_out, lh_out, hl_out, hh_out


class CrossBranchFusion(nn.Module):
    """Fuse spatial and wavelet features"""

    def __init__(self, channels):
        super().__init__()
        self.spatial_proj = nn.Conv2d(channels, channels, 1)
        self.wavelet_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, spatial_feat, wavelet_feat):
        if spatial_feat.shape[2:] != wavelet_feat.shape[2:]:
            wavelet_feat = F.interpolate(wavelet_feat, size=spatial_feat.shape[2:], mode="bilinear", align_corners=False)

        s = self.spatial_proj(spatial_feat)
        w = self.wavelet_proj(wavelet_feat)
        gate = self.gate(torch.cat([s, w], dim=1))
        fused = gate * s + (1 - gate) * w
        return self.out(fused)


# ======================== MAIN NETWORK ========================
class WaveFusionNet(nn.Module):
    """WaveFusion-Net: Dual-branch architecture for image deblurring"""

    def __init__(self, in_channels=3, out_channels=3, base_channels=48, num_blocks=(4, 6, 6, 4)):
        super().__init__()

        self.dwt = DWT()
        self.idwt = IDWT()

        self.intro = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.enc1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 2, stride=2)

        self.enc2 = nn.Sequential(*[NAFBlock(base_channels * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 2, stride=2)

        self.enc3 = nn.Sequential(*[NAFBlock(base_channels * 4) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 2, stride=2)

        self.wav_intro = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.wav_block1 = WaveletBlock(base_channels)
        self.wav_proj1 = nn.Conv2d(base_channels, base_channels * 2, 1)
        self.wav_block2 = WaveletBlock(base_channels * 2)
        self.wav_proj2 = nn.Conv2d(base_channels * 2, base_channels * 4, 1)
        self.wav_block3 = WaveletBlock(base_channels * 4)

        self.fusion1 = CrossBranchFusion(base_channels * 2)
        self.fusion2 = CrossBranchFusion(base_channels * 4)

        self.bottleneck = nn.Sequential(
            NAFBlock(base_channels * 8),
            StripAttention(base_channels * 8),
            NAFBlock(base_channels * 8),
            NAFBlock(base_channels * 8),
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(*[NAFBlock(base_channels * 4) for _ in range(num_blocks[2])])

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(*[NAFBlock(base_channels * 2) for _ in range(num_blocks[1])])

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(num_blocks[0])])

        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

        self.outro = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        f0 = self.intro(x)
        f1 = self.enc1(f0)
        f1_down = self.down1(f1)

        f2 = self.enc2(f1_down)
        f2_down = self.down2(f2)

        f3 = self.enc3(f2_down)
        f3_down = self.down3(f3)

        w0 = self.wav_intro(x)

        ll1, lh1, hl1, hh1 = self.dwt(w0)
        ll1, lh1, hl1, hh1 = self.wav_block1(ll1, lh1, hl1, hh1)
        w1 = self.wav_proj1(ll1)

        ll2, lh2, hl2, hh2 = self.dwt(w1)
        ll2, lh2, hl2, hh2 = self.wav_block2(ll2, lh2, hl2, hh2)
        w2 = self.wav_proj2(ll2)

        ll3, lh3, hl3, hh3 = self.dwt(w2)
        ll3, lh3, hl3, hh3 = self.wav_block3(ll3, lh3, hl3, hh3)

        f2_fused = self.fusion1(f2, w1)
        f3_fused = self.fusion2(f3, w2)

        bottleneck_out = self.bottleneck(f3_down)

        d3 = self.up3(bottleneck_out) + f3_fused
        d3 = self.dec3(d3)

        d2 = self.up2(d3) + f2_fused
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + f1
        d1 = self.dec1(d1)

        out = self.refine(d1)
        out = out + f0
        out = self.outro(out)
        out = out + x

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out
