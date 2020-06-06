from models.generators.GeneralGenerator import GeneralGenerator
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.generators.GeneralVAE import GeneralVAE
from models.losses.CalLoss import CalLoss
from models.losses.ComplexityLoss import ComplexityLoss

from models.losses.GeneralLoss import GeneralLoss
from models.losses.NonSaturatingGLoss import NonSaturatingGLoss
from models.losses.PixelLoss import PixelLoss

from utils.training_helpers import *
from typing import Tuple, Dict, Any

from utils.constants import *


class TotalGeneratorLoss(GeneralLoss):

    def __init__(self, NonSaturatingGLoss_weight = 1, PixelLoss_weight = 10, CalLoss_weight = 10, ComplexityLoss_weight: float = 10, **kwargs):

        super().__init__()

        self.adv = NonSaturatingGLoss(NonSaturatingGLoss_weight)
        self.pix = PixelLoss(PixelLoss_weight)
        self.cal = CalLoss(CalLoss_weight)
        self.kl = ComplexityLoss(ComplexityLoss_weight)


    def forward(self,
                images,
                labels,
                true_labelled,
                calnet_preds,
                calnet_labelled_imgs,
                preds,
                pred_labelled,
                pred_dist,
                pred_dist_labelled,
                generator: GeneralGenerator,
                discriminator: GeneralDiscriminator,
                args,
                b_index,
                dataset_size
                ) \
            -> Tuple[
                Any, Dict, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
        """ combined loss function for the generator """

        # check if we use a calibration net
        use_calibration_net = not args.calibration_net == "EmptyCalNet"

        # check that at least one loss function for the generator is active
        assert self.adv.active or self.pix.active or self.fm.active or self.cal.active or self.kl.active or self.pp.active or self.vq.active, "No generator loss function is active!"

        # compute total generator loss
        total_gen_loss = 0
        merged = {}

        # adverserial loss
        if self.adv.active:

            # merge batch and sample dimensions
            pd_shape = pred_dist_labelled.shape
            new_pd_shape = (-1, pd_shape[2], pd_shape[3], pd_shape[4])
            pred_dist_labelled = pred_dist_labelled.view(new_pd_shape)

            pred_score = discriminator(pred_dist_labelled)
            loss_adv, save_adv = self.adv(pred_score)

            merged = {**merged, **save_adv}
            total_gen_loss += loss_adv
            pred_labelled.detach()

        # pixel loss
        if self.pix.active:

            loss_pix, save_pix = self.pix(preds, labels)

            total_gen_loss += loss_pix
            merged = {**merged, **save_pix}

        if self.kl.active:
            assert isinstance(generator, GeneralVAE)

            mu, log_var = generator(images, return_mu_logvar=True) if not use_calibration_net else generator(calnet_labelled_imgs.detach(), return_mu_logvar=True)
            loss_kl, save_kl = self.kl(mu, log_var, b_index, dataset_size)
            total_gen_loss += loss_kl
            merged = {**merged, **save_kl}


        if self.cal.active:
            assert isinstance(generator, GeneralVAE), "Cal loss is only used when we have a vae generator/refinement net"
            assert use_calibration_net, "Cal loss is only used when we have a calibration net"

            avg_preds = pred_dist.mean(0)

            loss_cal, save_cal = self.cal(avg_preds, calnet_preds.detach(), labels, args)
            total_gen_loss += loss_cal
            merged = {**merged, **save_cal}


        return total_gen_loss, merged
