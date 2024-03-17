import os
import sys

import logging
import math

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .mobilenetV2 import MobileNetV2, InvertedResidual

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class MobileNetV2_unet(nn.Module):
    def __init__(self, n_classes=3, pre_trained=None):
        super(MobileNetV2_unet, self).__init__()

        # self.mode = mode
        self.backbone = MobileNetV2(n_channels=5, n_classes=3)

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)
        # ###!###
        self.dconv5 = nn.ConvTranspose2d(16, 8, 4, padding=1, stride=2)
        # self.invres5 = InvertedResidual(16, 8, 1, 6)  # inp, oup, stride, expand_ratio

        # self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_last = nn.Conv2d(8, n_classes, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)
        # self.conv_score = nn.Conv2d(n_channels, n_classes, 1)

        self._init_weights()

        if pre_trained is not None:
            # self.backbone.load_state_dict(torch.load(pre_trained, map_location="cpu"))
            self.backbone.load_state_dict(torch.load(pre_trained))

    def forward(self, x):
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x
        logging.debug((x1.shape, 'x1'))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        logging.debug((x2.shape, 'x2'))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        logging.debug((x3.shape, 'x3'))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        logging.debug((x4.shape, 'x4'))

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x
        logging.debug((x5.shape, 'x5'))

        up1 = torch.cat([
            x4,
            self.dconv1(x)
        ], dim=1)
        up1 = self.invres1(up1)
        logging.debug((up1.shape, 'up1'))

        up2 = torch.cat([
            x3,
            self.dconv2(up1)
        ], dim=1)
        up2 = self.invres2(up2)
        logging.debug((up2.shape, 'up2'))

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        up3 = self.invres3(up3)
        logging.debug((up3.shape, 'up3'))

        up4 = torch.cat([
            x1,
            self.dconv4(up3)
        ], dim=1)
        up4 = self.invres4(up4)
        logging.debug((up4.shape, 'up4'))

        up5 = self.dconv5(up4)
        logging.debug((up5.shape, 'up5'))

        x = self.conv_last(up5)
        logging.debug((x.shape, 'conv_last'))

        if not self.training:
            x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            logging.debug((x.shape, 'interpolate'))

        x = torch.sigmoid(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV2_unet(pre_trained=None)
    net(torch.randn(1, 3, 224, 224))
