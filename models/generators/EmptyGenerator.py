import torch

from models.generators.GeneralGenerator import GeneralGenerator
import torch.nn as nn


class EmptyGenerator(GeneralGenerator):
    """ for running without generator """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Note: Running without generator")


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