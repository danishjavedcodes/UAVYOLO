import torch
import torch.nn as nn

class AdaptiveSpatialAttention(nn.Module):
    """Adaptive Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(AdaptiveSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)

class AdaptiveChannelAttention(nn.Module):
    """Adaptive Channel Attention Module"""
    def __init__(self, channels, reduction_ratio=16):
        super(AdaptiveChannelAttention, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3))  # Global average pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

class HybridFeatureRefinementModule(nn.Module):
    """Hybrid Feature Refinement Module (HFRM) with multi-scale alignment"""
    def __init__(self, in_channels, out_channels):
        super(HybridFeatureRefinementModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        
        # Multi-scale pooling block (SPP-like) for better scale alignment
        self.spp_pool1 = nn.AdaptiveAvgPool2d(1)
        self.spp_pool3 = nn.AdaptiveAvgPool2d(3)
        self.spp_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        
        self.channel_attention = AdaptiveChannelAttention(out_channels)
        self.spatial_attention = AdaptiveSpatialAttention()
        self.amplification_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Multi-scale pooling for scale alignment
        spp_1 = self.spp_pool1(x)
        spp_3 = self.spp_pool3(x)
        spp_concat = torch.cat([spp_1, spp_3], dim=1)
        spp_out = self.spp_conv(spp_concat)
        x = x + spp_out  # Residual connection
        
        # Attention mechanisms for occlusion robustness
        x = self.channel_attention(x)
        x = self.spatial_attention(x) * x
        x = x * self.amplification_factor
        return x