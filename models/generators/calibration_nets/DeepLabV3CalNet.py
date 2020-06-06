import torch.nn as nn
import torch
import torch.nn.functional as F
from models.losses.NonSaturatingGLoss import NonSaturatingGLoss as GLoss
from utils.constants import CHANNEL_DIM, DEVICE
from models.generators.calibration_nets.GeneralCalNet import GeneralCalNet
from utils.pretrained_utils import resnet101_segmentation, deeplabv3_segmentation
from utils.training_helpers import renormalize


class DeepLabV3CalNet(GeneralCalNet):

    def __init__(self, n_channels_in: int = 3, n_channels_out: int = 3, n_hidden: int = 64,
                 norm_layer: nn.Module = nn.BatchNorm2d, use_dropout: bool = True,
                 n_hidden_layers: int = 3, padding_type: str = 'reflect', temperature: float = 1,
                 device: str = "cpu", **kwargs):

        super(DeepLabV3CalNet, self).__init__(n_channels_in, n_channels_out, device, **kwargs)

        # save for use in forward pass
        self.n_channels_in = n_channels_in
        self.temperature = temperature
        self.renormalize = lambda x: renormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model = deeplabv3_segmentation(n_classes=n_channels_out)
        self.model.backbone.conv1 = nn.Conv2d(n_channels_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    def forward(self, x, return_logits = False) -> torch.Tensor:

        if self.n_channels_in == 3:
            x = self.renormalize(x) # renormalize to normalization suggested by torchvision

        logits = self.model(x)['out']

        if return_logits:
            return logits
        else:
            return F.softmax(logits/self.temperature, dim=1)

if __name__ == '__main__':
    # Test if working

    dummy_batch = torch.rand((10, 3, 28, 28)).to(DEVICE)

    G = DeepLabV3CalNet(n_channels_out=3)

    gen_imgs = G(dummy_batch)

    print(gen_imgs.shape)

