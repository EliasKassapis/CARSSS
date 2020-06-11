from typing import Tuple, Dict, Optional
from utils.constants import *
import numpy as np
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision.utils import save_image

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_example_images(images, batches_done, suffix, filesort):
    """
    save some plots in PIC_DIR

    """
    normalize = True if images.max() > 1 and images.min() >= 0 else False

    save_image(images, f'./{PREFIX_OUTPUT}/{DATA_MANAGER.stamp}/{PIC_DIR}/{batches_done}_{suffix}.{filesort}',
               nrow=4, normalize=normalize)

def scheduler(i, D_number, G_number):

    if D_number == G_number == 1:
        return "DG"
    else:
        total = D_number + G_number
        if i % total < D_number:
            return "D"
        else:
            return "G"

def compute_accuracy(predictions, targets):
    """
    Gets the accuracy for discriminator

    """
    actual_predictions = predictions > 0.5
    true_positives = (actual_predictions == (targets > 0.5)).type(torch.DoubleTensor)
    accuracy = (torch.mean(true_positives))

    actual_predictions.detach()
    true_positives.detach()

    return accuracy.item()


def unpack_batch(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch.keys()) == 2 or (len(batch.keys()) == 3 and LABELS_CHANNELS == 25):
        return batch["image"].to(DEVICE).float(), batch["label"].to(DEVICE).float()
    elif len(batch.keys()) == 3 and LABELS_CHANNELS == 2:
        return batch["image"].to(DEVICE).float(), batch["label"].to(DEVICE).float(), batch["dist"].to(DEVICE).float()


def instance_checker(model, model_type):
    m = model.module if isinstance(model, DataParallel) else model
    return isinstance(m, model_type)


def renormalize(input_batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert input_batch.shape[1] == 3, f"Channel dimension C = {str(input_batch.shape[1])}. Expected C = 3"

    denorm_input = (input_batch + 1)/2

    shape = (1, 3, 1, 1)

    mean = torch.tensor(mean).reshape(shape).to(DEVICE)
    std = torch.tensor(std).reshape(shape).to(DEVICE)

    renorm_output = (denorm_input - mean)/std

    return renorm_output


def refactor_batch(input_batch, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=DEVICE):
    assert len(input_batch.shape) == 4

    output_batch = renormalize(torch.nn.functional.upsample_bilinear(input_batch, (size, size)).to(device), mean, std)

    return output_batch


def torch_comp_along_dim(funct, input: Optional, *args, multiple_idxs=False, dim=0):

    if type(input) == tuple:
        assert len(input) == 2, "Expected tuple of size: 2, received tuple of size: " + str(len(input))

        if multiple_idxs:
            return torch.stack([funct((input[0][i], input[1][i]), *args) for i in range(len(input[1]))], dim=dim)
        else:
            return torch.stack([funct((input[0], input[1][i]), *args) for i in range(len(input[1]))], dim=dim)
    else:
        if multiple_idxs:
            return torch.stack([funct(input[i], args[0][i], *args[1:]) for i in range(len(input))], dim=dim)
        else:
            return torch.stack([funct(input[i], *args) for i in range(len(input))], dim=dim)


def comp_along_dim(funct, input: Optional, *args, multiple_idxs=False, dim=0):

    if type(input) == tuple:
        assert len(input) == 2, "Expected tuple of size: 2, received tuple of size: " + str(len(input))

        if multiple_idxs:
            return [funct((input[0][i], input[1][i]), *args) for i in range(len(input[1]))]
        else:
            return [funct((input[0], input[1][i]), *args) for i in range(len(input[1]))]
    else:
        return [funct(input[i], *args) for i in range(len(input))]

def get_ce(pred: torch.Tensor, target: torch.Tensor, dim:int=CHANNEL_DIM)-> torch.Tensor:
    out = - target * torch.log(pred.clamp(min=1e-11)) # clamp to prevent gradient explosion
    return out.sum(dim)


def get_entropy(p: torch.Tensor, dim:int=CHANNEL_DIM) -> torch.Tensor:
    if type(p) != np.ndarray:
        out = -p * p.clamp(min=1e-7).log()
    else:
        out = -p * np.log(p.clip(min=1e-7))

    if dim == None:
        return out
    else:
        return out.sum(dim)


def tile(a, dim, n_tile):
    """
    This function is taken form PyTorch forum and mimics the behavior of tf.tile.
    Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(DEVICE)
    out= torch.index_select(a, dim, order_index)

    return out

def calibration_net_forward_pass(calibration_net, images, bb_preds, ign_idxs, args):

    if bb_preds is not None:
        p_input = torch.cat((images, bb_preds), dim=1)
    else:
        p_input = images

    calnet_preds_logits = calibration_net(p_input, return_logits=True)
    calnet_preds = F.softmax(calnet_preds_logits / args.temperature.reshape(1, LABELS_CHANNELS, 1, 1).to(DEVICE), dim=1)  # perform temperature scaling

    if ign_idxs is not None:
        # set unlabelled pixels to class unlabelled
        w = torch.ones(calnet_preds.shape)
        w[ign_idxs[0], :, ign_idxs[1], ign_idxs[2]] = 0.

        r = torch.zeros(calnet_preds.shape)
        r[ign_idxs[0], 24, ign_idxs[1], ign_idxs[2]] = 1.

        calnet_preds = calnet_preds * w.to(DEVICE) + r.to(DEVICE)

    calnet_labelled_images = torch.cat((images, calnet_preds.detach()), dim=CHANNEL_DIM)  # condition final prediction on input images and calibration net preds

    assert not torch.isnan(calnet_preds).any(), "Calibration net output is NaN"
    assert not torch.isinf(calnet_preds).any(), "Calibration net output is Inf"

    return calnet_preds_logits, calnet_preds, calnet_labelled_images


def generator_forward_pass(generator, images, calnet_labelled_images, ign_idxs, args):

    g_input = images if args.calibration_net == "EmptyCalNet" else calnet_labelled_images.detach()

    pred_dist,_,_ = generator.sample(g_input, ign_idxs=ign_idxs,  n_samples=args.n_cal_samples)
    pred_dist_labelled = torch_comp_along_dim(torch.cat, (images, pred_dist), CHANNEL_DIM, dim=0)

    assert not torch.isnan(pred_dist).any(), "Generator output is NaN"
    assert not torch.isinf(pred_dist).any(), "Generator output is Inf"

    preds = pred_dist[0]
    pred_labelled = pred_dist_labelled[0]


    return preds, pred_labelled, pred_dist, pred_dist_labelled


def discriminator_forward_pass(discriminator, true_labelled, pred_labelled, args):

    assert not args.generator == "EmptyGenerator", "Need to have an active generator to use a discriminator"

    # concat true and fake
    combined_input = torch.cat((pred_labelled.detach(), true_labelled.detach()), dim=0)  # todo should this be shuffled?

    # discriminator forward pass
    scores = discriminator(combined_input)

    assert not torch.isnan(scores).any(), "Discriminator output is NaN"
    assert not torch.isinf(scores).any(), "Discriminator output is Inf"


    shape = scores.shape
    shape = (shape[0]//2, *shape[1:])

    gt_labels = torch.cat((torch.zeros(shape).to(DEVICE), torch.ones(shape).to(DEVICE)), dim=0)

    # compute discriminator accuracy
    accuracy_discriminator = compute_accuracy(scores, gt_labels)

    return  combined_input, scores, gt_labels, accuracy_discriminator




