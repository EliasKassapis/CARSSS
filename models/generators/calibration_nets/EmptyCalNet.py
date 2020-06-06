import torch

from models.generators.calibration_nets.GeneralCalNet import GeneralCalNet
import torch.nn as nn


class EmptyCalNet(GeneralCalNet):
    """ for running without Calibration Net """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Note: Running without prior network")


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
