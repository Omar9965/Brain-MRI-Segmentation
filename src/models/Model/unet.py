import torch
import torch.nn as nn
from torch.nn.functional import relu


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_pool, max_pool], dim=1)
        out = self.conv(cat_out)
        M = torch.sigmoid(out)
        return x * M


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced = max(1, in_channels // reduction)
        self.avg_global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_channels, reduced, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Conv2d(reduced, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooling = self.avg_global_pooling(x)
        fc1 = self.fc(avg_pooling)
        relu = self.relu(fc1)
        fc2 = self.fc_out(relu)
        sigmoid = self.sigmoid(fc2)
        return x * sigmoid


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        x = self.channel_attention.forward(x)
        x = self.spatial_attention.forward(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = relu(x)
        x = self.dropout(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes=1, use_cbam=True, dropout_p=0.2):
        super(UNet, self).__init__()
        self.use_cbam = use_cbam

        # ENCODER PART
        self.enc1_1 = ConvBlock(3, 64, dropout_p=dropout_p)
        self.enc1_2 = ConvBlock(64, 64, dropout_p=dropout_p)
        self.cbam_enc1 = CBAM(64, reduction=16) if use_cbam else None
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2_1 = ConvBlock(64, 128, dropout_p=dropout_p)
        self.enc2_2 = ConvBlock(128, 128, dropout_p=dropout_p)
        self.cbam_enc2 = CBAM(128, reduction=16) if use_cbam else None
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3_1 = ConvBlock(128, 256, dropout_p=dropout_p)
        self.enc3_2 = ConvBlock(256, 256, dropout_p=dropout_p)
        self.cbam_enc3 = CBAM(256, reduction=16) if use_cbam else None
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_1 = ConvBlock(256, 512, dropout_p=dropout_p)
        self.enc4_2 = ConvBlock(512, 512, dropout_p=dropout_p)
        self.cbam_enc4 = CBAM(512, reduction=16) if use_cbam else None
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK
        self.bottleneck1 = ConvBlock(512, 1024, dropout_p=dropout_p * 1.5)
        self.bottleneck2 = ConvBlock(1024, 1024, dropout_p=dropout_p * 1.5)
        self.cbam_bottleneck = CBAM(1024, reduction=16) if use_cbam else None

        # DECODER PART
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1_1 = ConvBlock(1024, 512, dropout_p=dropout_p)
        self.dec1_2 = ConvBlock(512, 512, dropout_p=dropout_p)
        self.cbam_dec1 = CBAM(512, reduction=16) if use_cbam else None

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2_1 = ConvBlock(512, 256, dropout_p=dropout_p)
        self.dec2_2 = ConvBlock(256, 256, dropout_p=dropout_p)
        self.cbam_dec2 = CBAM(256, reduction=16) if use_cbam else None

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3_1 = ConvBlock(256, 128, dropout_p=dropout_p)
        self.dec3_2 = ConvBlock(128, 128, dropout_p=dropout_p)
        self.cbam_dec3 = CBAM(128, reduction=16) if use_cbam else None

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4_1 = ConvBlock(128, 64, dropout_p=dropout_p)
        self.dec4_2 = ConvBlock(64, 64, dropout_p=dropout_p)
        self.cbam_dec4 = CBAM(64, reduction=16) if use_cbam else None

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # ENCODER PART
        xe11 = self.enc1_1(x)
        xe12 = self.enc1_2(xe11)
        if self.use_cbam:
            xe12 = self.cbam_enc1(xe12)
        pool1 = self.pool1(xe12)

        xe21 = self.enc2_1(pool1)
        xe22 = self.enc2_2(xe21)
        if self.use_cbam:
            xe22 = self.cbam_enc2(xe22)
        pool2 = self.pool2(xe22)

        xe31 = self.enc3_1(pool2)
        xe32 = self.enc3_2(xe31)
        if self.use_cbam:
            xe32 = self.cbam_enc3(xe32)
        pool3 = self.pool3(xe32)

        xe41 = self.enc4_1(pool3)
        xe42 = self.enc4_2(xe41)
        if self.use_cbam:
            xe42 = self.cbam_enc4(xe42)
        pool4 = self.pool4(xe42)

        # BOTTLENECK
        xe51 = self.bottleneck1(pool4)
        xe52 = self.bottleneck2(xe51)
        if self.use_cbam:
            xe52 = self.cbam_bottleneck(xe52)

        # DECODER PART
        xu1 = self.up1(xe52)
        xu1 = torch.cat((xu1, xe42), dim=1)
        xu11 = self.dec1_1(xu1)
        xu12 = self.dec1_2(xu11)
        if self.use_cbam:
            xu12 = self.cbam_dec1(xu12)

        xu2 = self.up2(xu12)
        xu2 = torch.cat((xu2, xe32), dim=1)
        xu21 = self.dec2_1(xu2)
        xu22 = self.dec2_2(xu21)
        if self.use_cbam:
            xu22 = self.cbam_dec2(xu22)

        xu3 = self.up3(xu22)
        xu3 = torch.cat((xu3, xe22), dim=1)
        xu31 = self.dec3_1(xu3)
        xu32 = self.dec3_2(xu31)
        if self.use_cbam:
            xu32 = self.cbam_dec3(xu32)

        xu4 = self.up4(xu32)
        xu4 = torch.cat((xu4, xe12), dim=1)
        xu41 = self.dec4_1(xu4)
        xu42 = self.dec4_2(xu41)
        if self.use_cbam:
            xu42 = self.cbam_dec4(xu42)

        out = self.out(xu42)
        return out
