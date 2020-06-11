from models.losses.GeneralLoss import GeneralLoss
import torch
from utils.constants import DEVICE
import torch.nn.functional as F


class CalNetLoss(GeneralLoss):

    def __init__(self, CalNetLoss_weight=1, **kwargs):
        super(CalNetLoss, self).__init__(CalNetLoss_weight)

        self.weight = CalNetLoss_weight


    def custom_forward(self, calnet_preds_logits, labels, args):

        calnet_preds_train = F.softmax(calnet_preds_logits, dim=1)

        # if args.dataset == "LIDC":
        #     train_mask = torch.ones(labels.argmax(1).shape).to(DEVICE)
        # else:
        #     train_mask = (labels.argmax(1) != 24)
        #
        # loss = -((1*train_mask)*torch.sum(labels*torch.log(calnet_preds_train.clamp(min=1e-7)),dim=1)).mean()

        train_mask = (labels.argmax(1) != 24)
        ignore_mask = torch.ones(train_mask.shape).to(DEVICE)
        loss = -((1 * train_mask * ignore_mask) * torch.sum(labels * torch.log(calnet_preds_train.clamp(min=1e-7)),
                                                            dim=1)).mean()

        return self.weight*loss


if __name__ == '__main__':
    # Test if working
    dummy_batch = torch.rand((20, 3, 28, 28))

    dummy_ls1 = torch.rand((20, 68, 28, 28))
    dummy_ls2 = torch.rand((20, 68, 28, 28))

    get_loss = CalNetLoss()

    loss = get_loss(dummy_batch, dummy_ls1)[0]
    loss.backward()

    print(loss.item())
