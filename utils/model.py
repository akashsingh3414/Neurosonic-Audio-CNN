import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Improved residual block with proper initialization"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        out = out + shortcut
        out = torch.relu(out)
        return out


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AudioCNN(nn.Module):
    """Improved CNN for ESC-50 audio classification"""
    def __init__(self, num_classes=50):
        super().__init__()
        
        # Lighter stem - don't downsample too aggressively
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Reduced depth with attention
        self.layer1 = self._make_layer(32, 64, 2, stride=2)
        self.attention1 = ChannelAttention(64)
        
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.attention2 = ChannelAttention(128)
        
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.attention3 = ChannelAttention(256)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier with dropout
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256 * 2, num_classes)  # *2 for avg+max pooling
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, dropout=0.1))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=0.1))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_feature_maps=False):
        feature_maps = {}

        # Stem
        x = self.conv1(x)
        feature_maps["conv1"] = x
        
        # Layer 1 with attention
        x = self.layer1(x)
        x = self.attention1(x)
        feature_maps["layer1"] = x
        
        # Layer 2 with attention
        x = self.layer2(x)
        x = self.attention2(x)
        feature_maps["layer2"] = x
        
        # Layer 3 with attention
        x = self.layer3(x)
        x = self.attention3(x)
        feature_maps["layer3"] = x
        
        # Dual pooling (avg + max)
        avg_pool = self.avgpool(x)
        max_pool = self.maxpool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if return_feature_maps:
            return x, feature_maps
        return x
