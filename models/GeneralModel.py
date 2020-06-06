import torch.nn as nn


class GeneralModel(nn.Module):

    def __init__(self, n_input, device, **kwargs):
        self.n_input = n_input
        self.device = device

        super().__init__()

        if (len(kwargs) > 0):
            print(f"Ignored arguments in {self.__class__.__name__}: {kwargs}")
