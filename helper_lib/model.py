import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU()
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(c, c, 3, stride=2, padding=1), nn.SiLU())
    def forward(self, x): return self.op(x)

class Up(nn.Module):
    # 关键：指定 in/out 通道，先上采样，再卷到目标通道数，便于和 skip concat
    def __init__(self, in_c, out_c):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.SiLU()
        )
    def forward(self, x): return self.op(x)

class TinyUNet(nn.Module):
    def __init__(self, base=64, in_ch=3, out_ch=3):
        super().__init__()
        b = base
        # 编码器
        self.inc   = DoubleConv(in_ch, b)        # -> b
        self.down1 = Down(b);   self.dc1 = DoubleConv(b, b*2)    # -> 2b
        self.down2 = Down(b*2); self.dc2 = DoubleConv(b*2, b*4)  # -> 4b
        # 解码器
        self.up2 = Up(b*4, b*2)                    # 4b -> 2b
        self.uc2 = DoubleConv(b*2 + b*2, b*2)      # concat(2b,2b) -> 2b
        self.up1 = Up(b*2, b)                      # 2b -> b
        self.uc1 = DoubleConv(b + b, b)            # concat(b,b) -> b
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x, t=None):
        x1 = self.inc(x)                 # b
        x2 = self.dc1(self.down1(x1))    # 2b
        x3 = self.dc2(self.down2(x2))    # 4b
        u2 = self.up2(x3)                # 2b
        u2 = self.uc2(torch.cat([u2, x2], dim=1))  # 4b -> 2b
        u1 = self.up1(u2)                # b
        u1 = self.uc1(torch.cat([u1, x1], dim=1))  # 2b -> b
        return self.out(u1)              # 预测 ε

def build_diffusion_unet(): return TinyUNet(base=64)

# ---- Energy-Based Model (简单 CNN 给出 E(x)) ----
class SmallEBM(nn.Module):
    def __init__(self):
        super().__init__()
        C = 64
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1), nn.SiLU(),
            nn.Conv2d(C, C, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(C, C*2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(C*2, C*2, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(C*2, 1)
    def forward(self, x):
        h = self.net(x).flatten(1)
        e = self.fc(h)
        return e.squeeze(1)     # 标量能量

def build_ebm(): return SmallEBM()
