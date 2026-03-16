import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class ResNet50CBAM(nn.Module):
    """
    ResNet50 backbone enhanced with CBAM (Convolutional Block Attention Module).
    Forces the network to focus on grain structures rather than background tissue.
    """
    def __init__(self, pretrained=True):
        super(ResNet50CBAM, self).__init__()
        # Load standard ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Breakdown into stages to insert CBAM after each major residual block
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.cbam1 = CBAMBlock(256)
        
        self.layer2 = resnet.layer2
        self.cbam2 = CBAMBlock(512)
        
        self.layer3 = resnet.layer3
        self.cbam3 = CBAMBlock(1024)
        
        self.layer4 = resnet.layer4
        self.cbam4 = CBAMBlock(2048)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        return x # Returns feature map of shape [B, 2048, H/32, W/32]
