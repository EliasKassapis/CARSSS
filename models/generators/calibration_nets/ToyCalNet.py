import torch.nn as nn
import torch
import torch.nn.functional as F
from models.losses.NonSaturatingGLoss import NonSaturatingGLoss as GLoss
from utils.constants import CHANNEL_DIM
from models.generators.calibration_nets.GeneralCalNet import GeneralCalNet

class ToyCalNet(GeneralCalNet):
    """ Defines a Toy Generator"""

    def __init__(self, n_channels_in: int = 3, n_channels_out: int = 3, n_hidden: int = 64,
                 norm_layer: nn.Module = nn.BatchNorm2d, use_dropout: bool = True,
                 n_hidden_layers: int = 3, padding_type: str = 'reflect', temperature: float = 1,
                 device: str = "cpu", **kwargs):
        """
        n_channels_in (int)      - no. of channels in input images
        n_channels_out (int)     - no. number of channels in output images
        n_hidden (int)     - no. of filters in the last hidden layer
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        n_hidden_layers (int) - no of hidden layers
        padding_type (str) - type of padding: zero, replicate, or reflect
        temperature - magnitude of temperature scaling
        """
        super(ToyCalNet, self).__init__(n_channels_in, n_channels_out, device, **kwargs)

        # save for use in forward pass
        self.temperature = temperature

        # If normalizing layer is instance normalization, add bias
        # use_bias = norm_layer == nn.InstanceNorm2d
        use_bias = True
        use_dropout = True


        # Initialize model input block
        layers = []

        # Add input block layers
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(n_channels_in, n_hidden, kernel_size=3, bias=use_bias)]

        # layers += [nn.Linear(n_channels_in, n_hidden)]

        layers += [nn.Dropout(int(use_dropout) * 0.2)]
        layers += [norm_layer(n_hidden)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        # Add hidden block layers
        for i in range(n_hidden_layers):
            # Add input block layers
            layers += [nn.ReflectionPad2d(1)]
            layers += [nn.Conv2d(n_hidden, n_hidden, kernel_size=3, bias=use_bias)]

            # layers += [nn.Linear(n_hidden, n_hidden)]

            layers += [nn.Dropout(int(use_dropout) * 0.2)]
            layers += [norm_layer(n_hidden)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]

        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(n_hidden, n_channels_out, kernel_size=3)]

        # layers += [nn.Linear(n_hidden, n_channels_out)]

        layers += [nn.Dropout(int(use_dropout) * 0.2)]
        # layers += [nn.Softmax(dim=1)]

        # Save model
        self.model = nn.Sequential(*layers)

    def forward(self, x, return_logits = False) -> torch.Tensor:

        if return_logits:
            return self.model(x)
        else:
            return F.softmax(self.model(x)/self.temperature, dim=1) # todo pass args.temperature from prior initialization in main


if __name__ == '__main__':
    # Test if working

    dummy_batch = torch.rand((10, 3, 28, 28))

    G = ToyCalNet()

    gen_imgs = G(dummy_batch)