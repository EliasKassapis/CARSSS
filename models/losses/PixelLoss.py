from models.losses.GeneralLoss import GeneralLoss
from utils.training_helpers import get_ce


class PixelLoss(GeneralLoss):

    def __init__(self, weight=1, **kwargs):
        super(PixelLoss, self).__init__(weight)

    def custom_forward(self, pred, target, reduction = True):

        loss = get_ce(pred, target, dim=1)

        if reduction:
            return loss.mean()
        else:
            return loss