import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelReduce(nn.Module):
    """1x1 conv channel reduction."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsample + concat + double conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.skip_reduce = ChannelReduce(skip_ch, out_ch)
        self.up_reduce = ChannelReduce(in_ch, out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up_reduce(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.skip_reduce(skip)
        return self.conv(torch.cat([x, skip], dim=1))


class SegmentationDecoder(nn.Module):
    """UNet decoder with channel reduction."""
    def __init__(self, encoder_channels=(256, 512, 1024, 2048), num_classes=1):
        super().__init__()
        self.up4 = DecoderBlock(encoder_channels[3], encoder_channels[2], 256)
        self.up3 = DecoderBlock(256, encoder_channels[1], 128)
        self.up2 = DecoderBlock(128, encoder_channels[0], 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, skips):
        c1, c2, c3, c4 = skips
        x = self.up4(c4, c3)
        x = self.up3(x, c2)
        x = self.up2(x, c1)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.final(x))
