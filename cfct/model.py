import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channel, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class EdgeHead(nn.Module):
    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.out = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.out(x)


class FPNDecoderCBAM(nn.Module):
    def __init__(self, encoder_channels, out_channels=32):
        super().__init__()
        self.lateral4 = nn.Conv2d(encoder_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(encoder_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(encoder_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(encoder_channels[0], out_channels, 1)
        self.conv4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(), CBAM(out_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(), CBAM(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(), CBAM(out_channels))
        self.conv1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(), CBAM(out_channels))
        self.final = nn.Conv2d(out_channels, 1, 1)
        self.ds2 = nn.Conv2d(out_channels, 1, 1)
        self.ds3 = nn.Conv2d(out_channels, 1, 1)
        self.ds4 = nn.Conv2d(out_channels, 1, 1)
        self.edge_head = EdgeHead(out_channels)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=True)
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=True)
        p4 = self.conv4(p4)
        p3 = self.conv3(p3)
        p2 = self.conv2(p2)
        p1 = self.conv1(p1)
        out1 = self.final(F.interpolate(p1, scale_factor=4, mode='bilinear', align_corners=True))
        out2 = self.ds2(F.interpolate(p2, scale_factor=8, mode='bilinear', align_corners=True))
        out3 = self.ds3(F.interpolate(p3, scale_factor=16, mode='bilinear', align_corners=True))
        out4 = self.ds4(F.interpolate(p4, scale_factor=32, mode='bilinear', align_corners=True))
        edge_map = self.edge_head(p1)
        return out1, out2, out3, out4, edge_map


class HybridNet(nn.Module):
    def __init__(self, backbone_name='convnext_base', out_channels=32, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, pretrained=pretrained, out_indices=(0, 1, 2, 3))
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = FPNDecoderCBAM(encoder_channels, out_channels)

    def forward(self, x):
        feats = self.backbone(x)
        return self.decoder(feats)
