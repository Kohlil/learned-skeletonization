import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EfficientUNet5Down(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()

        # Contracting Path
        self.down1 = DoubleConv(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.down5 = DoubleConv(base_filters * 8, base_filters * 16)
        self.pool5 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_filters * 16, base_filters * 16)

        self.upconv5 = nn.ConvTranspose2d(
            base_filters * 16, base_filters * 16, kernel_size=2, stride=2
        )
        self.up5 = DoubleConv(base_filters * 16 + base_filters * 16, base_filters * 8)

        self.upconv4 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 8, kernel_size=2, stride=2
        )
        self.up4 = DoubleConv(base_filters * 8 + base_filters * 8, base_filters * 4)

        self.upconv3 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 4, kernel_size=2, stride=2
        )
        self.up3 = DoubleConv(base_filters * 4 + base_filters * 4, base_filters * 2)

        self.upconv2 = nn.ConvTranspose2d(
            base_filters * 2, base_filters * 2, kernel_size=2, stride=2
        )
        self.up2 = DoubleConv(base_filters * 2 + base_filters * 2, base_filters)

        self.upconv1 = nn.ConvTranspose2d(
            base_filters, base_filters, kernel_size=2, stride=2
        )
        self.up1 = DoubleConv(base_filters + base_filters, base_filters)

        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        d5 = self.down5(p4)
        p5 = self.pool5(d5)

        bottleneck = self.bottleneck(p5)

        u5 = self.upconv5(bottleneck)
        u5 = torch.cat([u5, d5], dim=1)
        u5 = self.up5(u5)

        u4 = self.upconv4(u5)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.up4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1)

        out = self.final_conv(u1)
        return out
