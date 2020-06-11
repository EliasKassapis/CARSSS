import torch
from torch import nn
from torchvision import models
from models.generators.calibration_nets.GeneralCalNet import GeneralCalNet
import torch.nn.functional as F
from utils.architecture_utils import initialize_weights

"""
Note: The code for the SegNet calibration net is adapted from https://github.com/zijundeng/pytorch-semantic-segmentation
"""

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels // 2

        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                      nn.Dropout(0.1),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNetCalNet(GeneralCalNet):
    def __init__(self, n_channels_in,n_channels_out, n_hidden, temperature = 1, device = "cpu", **kwargs):
        super(SegNetCalNet, self).__init__(n_channels_in, n_channels_out, device, **kwargs)
        self.temperature = temperature


        vgg = models.vgg19_bn(pretrained=True)

        vgg_features = list(vgg.features.children())

        if n_channels_in != 3:
            vgg_features[0] = nn.Conv2d(n_channels_in, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)).to(device)

        features = []
        for module in vgg_features:
            features.append(module)
            if isinstance(module, nn.ReLU):
                features.append(nn.Dropout(0.1))

        self.enc1 = nn.Sequential(*features[0:9])
        self.enc2 = nn.Sequential(*features[9:18])
        self.enc3 = nn.Sequential(*features[18:35])
        self.enc4 = nn.Sequential(*features[35:52])
        self.enc5 = nn.Sequential(*features[52:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True),
               nn.Dropout(0.1)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, n_channels_out, 2)

        self.out = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(n_channels_out, n_channels_out, 3, 1))

        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.out)

    def forward(self, x, return_logits = False):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1)) #skip connections
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))

        logits = self.out(dec1)

        if return_logits:
            return logits
        else:
            return F.softmax(logits/self.temperature, dim=1)

if __name__ == '__main__':
    imsize = (128,128)
    d_batch = torch.randn((14,20,*imsize))

    G = SegNetCalNet(n_channels_in=20, n_channels_out=17, n_hidden=64)

    print(G)

    pred = G(d_batch)
