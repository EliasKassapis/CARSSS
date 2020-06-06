import torch
from torch import nn
import torch.nn.functional as F
from utils.architecture_utils import initialize_weights, Noise_injector
from models.generators.GeneralVAE import GeneralVAE
from utils.training_helpers import torch_comp_along_dim
from utils.constants import DEVICE
from torch.distributions import Normal, Independent

class UNetGenerator(GeneralVAE):
    def __init__(self, imsize, n_channels_in,n_channels_out, n_hidden, z_dim, device = "cpu", **kwargs):
        super(UNetGenerator, self).__init__(n_channels_in, n_channels_out, device, **kwargs)

        self.z_dim = z_dim

        hidden_dims = [n_hidden, n_hidden*2, n_hidden*4, n_hidden*8]

        # embedder
        self.enc1 = nn.Sequential(nn.Conv2d(n_channels_in, hidden_dims[0], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(hidden_dims[0]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.enc2 = nn.Sequential(nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(hidden_dims[1]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.enc3 = nn.Sequential(nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(hidden_dims[2]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.enc4 = nn.Sequential(nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(hidden_dims[3]), nn.LeakyReLU(0.2), nn.Dropout(0.1))

        self.dec0 = nn.Sequential(nn.ConvTranspose2d(hidden_dims[3], hidden_dims[2], kernel_size=2, stride=2), nn.BatchNorm2d(hidden_dims[2]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(hidden_dims[3], hidden_dims[1], kernel_size=2, stride=2), nn.BatchNorm2d(hidden_dims[1]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(hidden_dims[2], hidden_dims[0], kernel_size=2, stride=2), nn.BatchNorm2d(hidden_dims[0]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(hidden_dims[1],hidden_dims[0], kernel_size=2, stride=2), nn.BatchNorm2d(hidden_dims[0]), nn.LeakyReLU(0.2), nn.Dropout(0.1))

        self.zres1 = Noise_injector(hidden_dims[1], z_dim, n_channels_in, hidden_dims[1], device=device).to(device)
        self.zres2 = Noise_injector(hidden_dims[0], z_dim, n_channels_in, hidden_dims[0], device=device).to(device)
        self.out = Noise_injector(hidden_dims[0], z_dim, n_channels_in, n_channels_out, device=device).to(device)

        initialize_weights(self.dec3, self.dec2, self.dec1, self.dec0)

    def forward(self, x):

        self.encode(x)

        self.get_gauss(x)

        z = self.gauss.sample()

        return self.decode(z)


    def encode(self, x):
        self.down1 = self.enc1(x)
        self.down2 = self.enc2(self.down1)
        self.down3 = self.enc3(self.down2)
        self.down4 = self.enc4(self.down3)

    def decode(self, z, ign_idxs=None):
        up1 = self.dec0(self.down4)
        up2 = self.dec1(torch.cat((up1, self.down3),dim=1)) #skip connection

        up2b = nn.functional.leaky_relu(self.zres1(up2, z)) # noise injection

        up3 = self.dec2(torch.cat((up2b, self.down2), dim=1))

        up3b = nn.functional.leaky_relu(self.zres2(up3, z))

        up4 = self.dec3(torch.cat((up3b, self.down1),dim=1))

        logits = self.out(up4,z)

        out =  F.softmax(logits, dim=1)

        if ign_idxs is None:
            return out
        else:
            # set unlabelled pixels to class unlabelled for Cityscapes
            # masks the adv loss by preventing gradients from being formed in unlabelled pixs
            w = torch.ones(out.shape)
            w[ign_idxs[0], :, ign_idxs[1], ign_idxs[2]] = 0.

            r = torch.zeros(out.shape)
            r[ign_idxs[0], 24, ign_idxs[1], ign_idxs[2]] = 1.

            out = out * w.to(DEVICE) + r.to(DEVICE)

            return out

    def get_gauss(self, x):
        b_size = len(x)
        self.gauss = Independent(Normal(loc=torch.zeros((b_size, self.z_dim)).float().to(DEVICE),
                                        scale=torch.ones((b_size, self.z_dim)).float().to(DEVICE)), 1)

    def sample(self, x, ign_idxs = None, n_samples=1):

        self.get_gauss(x)

        # sample z
        z = self.gauss.sample((n_samples,))

        # encode z
        self.encode(x)

        # serial decoding
        if ign_idxs is None:
            pred_dist = torch_comp_along_dim(self.decode, z, dim=0)
        else:
            pred_dist = torch_comp_along_dim(self.decode, z, ign_idxs, dim=0)

        # compute the average prediction
        avg_pred = pred_dist.mean(0)

        return pred_dist, None, avg_pred


if __name__ == '__main__':
    imsize = (128,128)
    d_batch = torch.randn((14,5,*imsize)).to("cuda")

    G = UNetGenerator(imsize = imsize, n_channels_in=5, n_channels_out=2, n_hidden=64, z_dim=32, device="cuda").to("cuda")

    print(G)

    pred = G(d_batch)

    print(pred.shape)