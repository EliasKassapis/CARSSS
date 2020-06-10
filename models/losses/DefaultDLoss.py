from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from utils.constants import DEVICE

class DefaultDLoss(GeneralLoss):

    def __init__(self, weight, **kwargs):
        super(DefaultDLoss, self).__init__(weight=weight)


    def custom_forward(self, fake_scores, real_scores, smoothing=False):

        get_loss = nn.BCELoss()

        fake_labels = torch.zeros(fake_scores.shape).to(DEVICE)
        real_labels = torch.ones(real_scores.shape).to(DEVICE)

        # label smoothing
        if smoothing:
            fake_labels = torch.FloatTensor(fake_labels.shape).uniform_(0, 0.3).to(DEVICE)
            real_labels = torch.FloatTensor(real_labels.shape).uniform_(0.7, 1).to(DEVICE)


        loss = get_loss(fake_scores, fake_labels).mean() + get_loss(real_scores, real_labels).mean()

        return loss


