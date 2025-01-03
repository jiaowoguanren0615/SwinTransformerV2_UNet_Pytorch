import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.mp = nn.AdaptiveAvgPool2d((1, 1))
        self.ap = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.mp(x))
        avg_out = self.fc(self.ap(x))
        out = avg_out + max_out
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv1(x))
        return y


class CbamModule(nn.Module):
    def __init__(
            self, channels, rd_ratio=16, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttention(channels, rd_ratio)
        self.spatial = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x