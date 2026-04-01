# model.py
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DepthwiseSeparableConv(1,  16)
        self.enc2 = DepthwiseSeparableConv(16, 32)
        self.enc3 = DepthwiseSeparableConv(32, 64)
        self.bot  = DepthwiseSeparableConv(64, 64)
        self.dec1 = DepthwiseSeparableConv(128, 32)
        self.dec2 = DepthwiseSeparableConv(64,  16)
        self.dec3 = DepthwiseSeparableConv(32,  16)
        self.out  = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bot(e3)
        d  = self.dec1(torch.cat([b,  e3], 1))
        d  = self.dec2(torch.cat([d,  e2], 1))
        d  = self.dec3(torch.cat([d,  e1], 1))
        # Residual: model predicts correction on top of noisy input
        return torch.relu(x + self.out(d))


if __name__ == "__main__":
    m = DenoiseNet()
    print(f"Parameters: {sum(p.numel() for p in m.parameters()):,}")
    x = torch.rand(1, 1, 257, 200)
    print(f"Output shape: {m(x).shape}")
    print("OK ✓")
