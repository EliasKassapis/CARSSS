from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch

from utils.architecture_utils import Flatten


class PixelDiscriminator(GeneralDiscriminator):

    def __init__(self,imsize, n_channels_in=3, n_hidden=64, n_layers=3,use_dropout: bool=False,
                 device: str="cpu", **kwargs):

        super(PixelDiscriminator, self).__init__(n_channels_in, device, **kwargs)

        # If normalizing layer is batch normalization, don't add bias because nn.BatchNorm2d has affine params
        use_bias = False
        layers = []
        out = []

        # Add input block auditor
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(n_channels_in, n_hidden, kernel_size=3, stride=1, bias=True)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        # Add hidden blocks auditor
        for i in range(2):
            layers += [nn.ReflectionPad2d(1)]
            layers += [nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, bias=True)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]

        layers += [nn.ReflectionPad2d(1)]

        layers += [nn.Conv2d(n_hidden, n_hidden, kernel_size=4, stride=2, padding=1)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Dropout(0.1 * int(use_dropout))]

        # Set factor of change for input and output channels for hidden layers
        mult_in = 1
        mult_out = 1

        # Add hidden layers
        for i in range(1, n_layers + 1):

            mult_in = mult_out
            mult_out = min(2 ** i, 8)

            if i == n_layers:
                layers += [nn.Conv2d(n_hidden * mult_in, n_hidden * mult_out, kernel_size=4, stride=1, padding=1,
                                     bias=use_bias)]  # stride = 1
            else:
                layers += [nn.Conv2d(n_hidden * mult_in, n_hidden * mult_out, kernel_size=4, stride=2, padding=1,
                                     bias=use_bias)]  # stride = 2

            layers += [nn.LeakyReLU(0.2, inplace=True)]
            layers += [nn.Dropout(0.1*int(use_dropout))]

        #  output layer (1 channel prediction map)
        layers += [nn.Conv2d(n_hidden * mult_out, n_hidden, kernel_size=4, stride=1, padding=1)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Dropout(0.1*int(use_dropout))]


        out += [Flatten()]
        out += [nn.Linear(n_hidden * ((imsize[0]//2**n_layers) - 2) * ((imsize[1]//2**n_layers) - 2), 1)]

        out += [nn.Dropout(0.1*int(use_dropout))]
        out += [nn.Sigmoid()]

        # Save model
        self.model = nn.Sequential(*layers)
        self.out = nn.Sequential(*out)

    def forward(self, x) -> torch.Tensor:

        assert not torch.isnan(x).any(), "Discriminator input is NaN"
        assert not torch.isinf(x).any(), "Discriminator input is inf"

        feats = self.model(x)

        y = self.out(feats).clamp(min=1e-7)

        assert not torch.isnan(y).any(), "Discriminator output is NaN"
        assert not torch.isinf(y).any(), "Discriminator output is inf"

        return y.squeeze()


if __name__ == '__main__':
    # Test if working

    dummy_batch = torch.rand((32, 71, 256, 256))

    D = PixelDiscriminator(imsize=(256, 256), n_channels_in=71)

    score = D(dummy_batch)
