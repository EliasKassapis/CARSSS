from models.losses.GeneralLoss import GeneralLoss
from utils.constants import *

class CalLoss(GeneralLoss):

    def __init__(self, weight: float, **kwargs):
        super(CalLoss, self).__init__(weight=weight)

    def custom_forward(self, avg_preds, prior_preds, labels, args):

        eps = 1e-3

        # mask loss to train only on labelled idxs
        if args.dataset == "LIDC":
            train_mask = torch.ones(labels.argmax(1).shape).to(DEVICE)
        else:
            train_mask = (labels.argmax(1) != 24)

        kl = lambda p, q: (-p.clamp(min=eps, max=1 - eps) * torch.log(q.clamp(min=eps, max=1 - eps))
                           + p.clamp(min=eps, max=1 - eps) * torch.log(p.clamp(min=eps, max=1 - eps))).sum(1)

        loss = ((1 * train_mask)*kl(avg_preds, prior_preds)).mean() # rKL where avg preds is used as target p

        return loss