"""
PyTorch model definitions — mirrors the notebook so checkpoints load correctly.
Supports both the baseline SmallUNet and the improved AttentionUNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Baseline SmallUNet ────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class SmallUNet(nn.Module):
    def __init__(self, base_ch=48):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResidualSEBlock(1, base_ch)
        self.enc2 = ResidualSEBlock(base_ch, base_ch * 2, dropout=0.05)
        self.enc3 = ResidualSEBlock(base_ch * 2, base_ch * 4, dropout=0.1)
        self.enc4 = ResidualSEBlock(base_ch * 4, base_ch * 8, dropout=0.15)
        self.bottleneck = ResidualSEBlock(base_ch * 8, base_ch * 16, dropout=0.2)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ResidualSEBlock(base_ch * 16, base_ch * 8, dropout=0.1)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ResidualSEBlock(base_ch * 8, base_ch * 4, dropout=0.1)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ResidualSEBlock(base_ch * 4, base_ch * 2, dropout=0.05)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ResidualSEBlock(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ── Improved AttentionUNet ────────────────────────────────────────────

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class ResidualSEBlockV2(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        hidden = max(out_ch // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1), nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        return self.relu(out + identity)


class AttentionUNet(nn.Module):
    def __init__(self, base_ch=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResidualSEBlockV2(1, ch[0])
        self.enc2 = ResidualSEBlockV2(ch[0], ch[1], dropout=0.05)
        self.enc3 = ResidualSEBlockV2(ch[1], ch[2], dropout=0.1)
        self.enc4 = ResidualSEBlockV2(ch[2], ch[3], dropout=0.15)
        self.bottleneck = ResidualSEBlockV2(ch[3], ch[4], dropout=0.2)
        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], 2, stride=2)
        self.ag4 = AttentionGate(ch[3], ch[3], ch[3] // 2)
        self.dec4 = ResidualSEBlockV2(ch[3] * 2, ch[3], dropout=0.1)
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.ag3 = AttentionGate(ch[2], ch[2], ch[2] // 2)
        self.dec3 = ResidualSEBlockV2(ch[2] * 2, ch[2], dropout=0.1)
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.ag2 = AttentionGate(ch[1], ch[1], ch[1] // 2)
        self.dec2 = ResidualSEBlockV2(ch[1] * 2, ch[1], dropout=0.05)
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.ag1 = AttentionGate(ch[0], ch[0], ch[0] // 2)
        self.dec1 = ResidualSEBlockV2(ch[0] * 2, ch[0])
        self.head = nn.Conv2d(ch[0], 1, 1)
        if deep_supervision:
            self.aux4 = nn.Conv2d(ch[3], 1, 1)
            self.aux3 = nn.Conv2d(ch[2], 1, 1)
            self.aux2 = nn.Conv2d(ch[1], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        g4 = self.up4(b);  d4 = self.dec4(torch.cat([g4, self.ag4(g4, e4)], 1))
        g3 = self.up3(d4); d3 = self.dec3(torch.cat([g3, self.ag3(g3, e3)], 1))
        g2 = self.up2(d3); d2 = self.dec2(torch.cat([g2, self.ag2(g2, e2)], 1))
        g1 = self.up1(d2); d1 = self.dec1(torch.cat([g1, self.ag1(g1, e1)], 1))
        logits = self.head(d1)
        if self.deep_supervision and self.training:
            sz = x.shape[2:]
            return (
                logits,
                F.interpolate(self.aux4(d4), sz, mode="bilinear", align_corners=False),
                F.interpolate(self.aux3(d3), sz, mode="bilinear", align_corners=False),
                F.interpolate(self.aux2(d2), sz, mode="bilinear", align_corners=False),
            )
        return logits
