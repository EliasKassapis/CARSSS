from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from utils.constants import DEVICE


class NonSaturatingGLoss(GeneralLoss):

    def __init__(self, weight, **kwargs):
        super(NonSaturatingGLoss, self).__init__(weight=weight)

    def custom_forward(self, score: torch.Tensor):

        get_loss = nn.BCELoss()

        loss = get_loss(score, torch.ones(score.shape).to(DEVICE))

        return loss
