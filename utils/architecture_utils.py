import torch.nn as nn

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights_unet(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):

    def __init__(self, n_channels, im_size):
        self.n_channels = n_channels
        self.im_size = im_size
        super().__init__()

    def forward(self, input):
        out =  input.view(input.size(0), self.n_channels, self.im_size[0], self.im_size[1])
        return out

class Noise_injector(nn.Module):
    def __init__(self, n_hidden, z_dim, num_channels, n_channels_out, device="cpu"):

        super(Noise_injector, self).__init__()
        self.num_channels = num_channels #output channels
        self.n_channels_out = n_channels_out
        self.n_hidden = n_hidden
        self.z_dim = z_dim
        self.device = device

        self.residual = nn.Linear(self.z_dim, self.n_hidden)
        self.scale = nn.Linear(self.z_dim, self.n_hidden)

        self.last_layer = nn.Conv2d(self.n_hidden, self.n_channels_out, kernel_size=1)

        self.residual.apply(weights_init)
        self.scale.apply(weights_init)
        self.last_layer.apply(init_weights_orthogonal_normal)

    def forward(self, feature_map, z):
        """
        Z is B x Z_dim and feature_map is B x C x H x W.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """

        # affine transf
        residual = self.residual(z).view(z.shape[0], self.n_hidden, 1, 1)
        scale = self.scale(z).view(z.shape[0], self.n_hidden, 1, 1)

        feature_map = (feature_map + residual) * (scale + 1e-5)

        return self.last_layer(feature_map)

