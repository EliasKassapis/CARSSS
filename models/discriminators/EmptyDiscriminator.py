import torch
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator


class EmptyDiscriminator(GeneralDiscriminator):
    """ for running without a discriminator """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Note: Running without a discriminator")


    @staticmethod
    def forward(_):
        return None

    @staticmethod
    def parameters(*args):
        return [torch.LongTensor([])]

    @staticmethod
    def state_dict(**kwargs):
        return None

    @staticmethod
    def load_state_dict(self, *args, **kwargs):
        pass