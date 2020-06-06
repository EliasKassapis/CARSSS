from models.losses.DefaultDLoss import DefaultDLoss
from models.losses.GeneralLoss import GeneralLoss


from utils.training_helpers import *
from typing import Tuple, Dict, Any

from utils.constants import *


class TotalDiscriminatorLoss(GeneralLoss):

    def __init__(self, DefaultDLoss_weight=1,**kwargs):

        super().__init__()

        self.default = DefaultDLoss(DefaultDLoss_weight)


    def forward(self,
                discriminator,
                combined_input,
                scores,
                gt_labels,
                args,
                ):
        """ combined loss function for the discriminator """

        # initialize total discriminator loss
        total_disc_loss = 0
        merged = {}

        split_idx = combined_input.shape[0]//2

        predictions = combined_input[:split_idx]
        fake_scores = scores[:split_idx]

        labels = combined_input[split_idx:]
        real_scores = scores[split_idx:]


        if self.default.active:
            loss_default, save_default = self.default(fake_scores, real_scores, smoothing = args.label_smoothing)
            total_disc_loss += loss_default
            merged = {**merged, **save_default}

        return (total_disc_loss, merged)


if __name__ == '__main__':
    loss_func = TotalDiscriminatorLoss()
