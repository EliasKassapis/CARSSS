import wandb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision.utils import save_image
from utils.data_utils import _recolour_label, _1hot_2_2d, de_torch, move_color_channel
from utils.training_helpers import instance_checker, unpack_batch
from models.generators.GeneralVAE import GeneralVAE
from utils.constants import *
from pathlib import Path
import sklearn.metrics as metrics
import scipy

"""
Note: Many functions used here are adapted from the Probabilistic U-Net re-implementation repository at https://github.com/SimonKohl/probabilistic_unet
"""


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0

    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def get_foreground_IoU(preds, labels):

    # convert labels and predictions to discrete format
    preds = preds.argmax(0).flatten().cpu().numpy()

    labels = labels.argmax(0).flatten().cpu().numpy()

    if preds.sum() + labels.sum() == 0:
        return 1.0
    elif labels.sum() > 0 and preds.sum() == 0 or preds.sum() == 0 and labels.sum() > 0:
        return 0.
    else:
        return metrics.jaccard_score(preds, labels)


def get_confusion_matrix(preds, labels, class_idxs = None, ignore=24):
    try:
        assert labels.shape == preds.shape
    except:
        raise AssertionError(f'shape mismatch {labels.shape} vs. {preds.shape}')

    if class_idxs is None:
        n_labels = LABELS_CHANNELS
        classes = torch.arange(n_labels).to(DEVICE)
    else:
        n_labels = len(class_idxs)
        classes = torch.tensor(class_idxs)

    # initialize confusion matrix
    c_matrix = torch.zeros((n_labels, 4)).to(DEVICE)

    # convert labels and predictions to discrete format
    preds = preds.argmax(0).to(DEVICE)
    labels = labels.argmax(0).to(DEVICE)

    # ignore mask
    if ignore is None:
        shape = labels.shape
        ignore_mask = torch.zeros((shape[0], 1, shape[2], shape[3]))
    else:
        ignore_mask = (labels == 24)

    for i,c in enumerate(classes):

        b_preds = (preds == c) # binarize predictions
        b_labels_ = (labels == c) # binarize labels

        c_matrix[i,0] = int(((b_preds != 0) * (b_labels_ != 0) * (ignore_mask != 1)).sum()) # TP
        c_matrix[i,1] = int(((b_preds != 0) * (b_labels_ == 0) * (ignore_mask != 1)).sum()) # FP
        c_matrix[i,2] = int(((b_preds == 0) * (b_labels_ == 0) * (ignore_mask != 1)).sum()) # TN
        c_matrix[i,3] = int(((b_preds == 0) * (b_labels_ != 0) * (ignore_mask != 1)).sum()) # FN

    return c_matrix

def get_IoU(c_matrix):
    tp = c_matrix[:, 0]
    fp = c_matrix[:, 1]
    fn = c_matrix[:, 3]

    iou = torch.zeros(tp.shape).to(DEVICE)

    classes = torch.arange(tp.shape[0])

    for c in classes:
        # unless both the prediction and the ground-truth is empty, calculate a finite IoU
        if tp[c] + fp[c] + fn[c] != 0:
            iou[c] = tp[c] / (tp[c] + fp[c] + fn[c])
        else:
            iou[c] = np.nan

    return iou

def get_mode_statistics(probabilities, n_flipped_modes=5):

    num_modes = 2 ** n_flipped_modes

    # assemble a binary matrix of flips decisions
    flips = np.zeros(shape=(num_modes, 5), dtype=np.uint8)
    for i in range(n_flipped_modes):
        flips[:,i] = 2 ** i * (2 ** (n_flipped_modes - 1 - i) * [0] + 2 ** (n_flipped_modes - 1 - i) * [1]) # gets all mode permutations

    # calculate the probability for each individual mode
    mode_probs = torch.zeros((num_modes,)).float().to(DEVICE)
    for mode in range(num_modes):
        prob = 1.
        for i, p in enumerate(probabilities):
            if flips[mode, i]:
                prob *= p
            else:
                prob *= 1. - p
        mode_probs[mode] = torch.tensor(prob)
    assert torch.sum(mode_probs) == 1.

    return {'flips': flips, 'probs': mode_probs}


def get_cs_ignore_mask(bb_preds, labels):

    assert bb_preds.shape == labels.shape, f"bb_preds shape = {bb_preds.shape}, labels shape = {labels.shape}"

    orig_classes = torch.arange(25)
    orig_classes[19:24] = torch.tensor([1, 11, 13, 8, 0])

    # convert all flipped classes to original classes
    pres = orig_classes[bb_preds.argmax(1)]
    lbs = orig_classes[labels.argmax(1)]
    mask = pres == lbs

    return mask


def get_energy_distance_components(labels, preds, class_idxs = None, n_labels = None, args=None):

    if class_idxs is None:
        n_classes = LABELS_CHANNELS
    else:
        n_classes = len(class_idxs)

    if n_labels is None:
        n_labels = labels.shape[0]

    n_present_labels = labels.shape[0]
    n_preds = preds.shape[0]

    d_matrix_YS = torch.zeros((n_present_labels, n_preds, n_classes)).to(DEVICE)
    d_matrix_YY = torch.zeros((n_labels, n_labels, n_classes)).to(DEVICE)
    d_matrix_SS = torch.zeros((n_preds, n_preds, n_classes)).to(DEVICE)

    for j in range(n_present_labels):
        for i in range(n_preds):

            if args.dataset == "CITYSCAPES19":
                c_matrix = get_confusion_matrix(labels[j], preds[i], class_idxs=class_idxs)
                iou = get_IoU(c_matrix)
            else:
                iou = get_foreground_IoU(labels[j], preds[i])

            d_matrix_YS[j, i] = 1. - iou

        if j <= n_labels:
            for not_j in range(j, n_labels):
                if args.dataset == "CITYSCAPES19":
                    c_matrix = get_confusion_matrix(labels[j], labels[not_j], class_idxs=class_idxs)
                    iou = get_IoU(c_matrix)
                else:
                    iou = get_foreground_IoU(labels[j], labels[not_j])

                d_matrix_YY[j, not_j] = 1. - iou
                d_matrix_YY[not_j, j] = 1. - iou

    for i in range(n_preds):
        # iterate all samples S'
        for j in range(i, n_preds):
            if args.dataset == "CITYSCAPES19":
                c_matrix = get_confusion_matrix(preds[i], preds[j], class_idxs=class_idxs)
                iou = get_IoU(c_matrix)
            else:
                iou = get_foreground_IoU(preds[i], preds[j])

            d_matrix_SS[i, j] = 1. - iou
            d_matrix_SS[j, i] = 1. - iou

    return {'YS': d_matrix_YS, 'SS': d_matrix_SS, 'YY': d_matrix_YY}


def calc_energy_distances(d_matrices, num_samples=None, flip_probs = None):

    d_matrices = d_matrices.copy()

    if num_samples is None:
        num_samples = d_matrices['SS'].shape[1]

    d_matrices['YS'] = d_matrices['YS'][:,:,:num_samples]
    d_matrices['SS'] = d_matrices['SS'][:,:num_samples,:num_samples]

    if flip_probs is not None:
        mode_stats = get_mode_statistics(probabilities=flip_probs, n_flipped_modes=len(flip_probs))
        mode_probs = mode_stats['probs']

        mean_d_YS = nanmean(d_matrices['YS'], dim=-1)
        mean_d_YS = torch.mean(mean_d_YS, dim=2)
        mean_d_YS = mean_d_YS * mode_probs[np.newaxis, :]
        d_YS = torch.sum(mean_d_YS, dim=1)

        mean_d_SS = nanmean(d_matrices['SS'], dim=-1)
        d_SS = torch.mean(mean_d_SS, dim=(1, 2))

        mean_d_YY = nanmean(d_matrices['YY'], dim=-1)
        mean_d_YY = mean_d_YY * mode_probs[np.newaxis, :, np.newaxis] * mode_probs[np.newaxis, np.newaxis, :]
        d_YY = torch.sum(mean_d_YY, dim=(1, 2))

    else:

        mean_d_YS = nanmean(d_matrices['YS'], dim=-1)

        d_YS = mean_d_YS.mean(dim=(1,2))

        mean_d_SS = nanmean(d_matrices['SS'], dim=-1)
        d_SS = mean_d_SS.mean(dim=(1,2))

        mean_d_YY = nanmean(d_matrices['YY'], dim=-1)
        d_YY = mean_d_YY.mean(dim=(1,2))

    GED = 2 * d_YS - d_SS - d_YY

    return GED, d_YS, d_SS

def compute_iou(fake_labels, labels, args):

    if args.dataset == "CITYSCAPES19":
        c_matrix = get_confusion_matrix(fake_labels, labels)
        iou = get_IoU(c_matrix)
    else:
        iou = get_foreground_IoU(labels, fake_labels)

    return iou


def compute_ged(pred_dist, gt_dist, calnet_preds, args, g_input = None, n_samples=20, return_hungarian = False):

    orig_n_labels = None

    if args.dataset == "LIDC":
        eval_class_idxs = None
        probs = None

        # convert labels from discrete to 1-hot format and move channel dimension
        lab_dist = torch.eye(LABELS_CHANNELS)[(gt_dist).long()].permute(1,0,4,2,3).to(DEVICE)

        # todo if eval hungarian
        # duplicate gt labels as many times as it is needed so that the number of elements in
        # the predictive distribution and gt distribution are equal
        if return_hungarian and len(lab_dist)!=args.n_generator_samples_test:
            orig_n_labels = len(lab_dist)
            n_repeats = args.n_generator_samples_test//len(lab_dist)
            lab_dist = lab_dist.repeat(n_repeats,1,1,1,1)
            assert len(lab_dist)==args.n_generator_samples_test, f"Lab dist ({len(lab_dist)}) and pred samples ({args.n_generator_samples_test}) need to be the same number to compute the hungarian-matched iou. "

    elif args.dataset == "CITYSCAPES19":

        lab_dist = torch.eye(LABELS_CHANNELS)[(gt_dist).long()].permute(1, 0, 4, 2, 3)

        # get the idxs of switchable classes and the probability of each flip
        eval_class_idxs = eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")[0] + eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")[1]
        probs = eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")[2]

    d_matrices = {'YS': [], 'SS': [], 'YY': []}
    if return_hungarian: hungarian_scores = [] # initialize variable

    # aggregate results
    for i in range(lab_dist.shape[1]):
        d_matrix = get_energy_distance_components(lab_dist[:,i,:,:,:], pred_dist[:,i,:,:,:], class_idxs=eval_class_idxs, n_labels = orig_n_labels, args=args)

        if return_hungarian:
            assert len(lab_dist)==args.n_generator_samples_test, f"Lab dist ({len(lab_dist)}) and pred samples ({args.n_generator_samples_test}) need to be the same number to compute the hungarian-matched iou. "

            # compute Hungarian-matched IoU
            cost_matrix = nanmean(d_matrix['YS'], dim=-1)
            h_score = (1-cost_matrix)[scipy.optimize.linear_sum_assignment(cost_matrix.cpu().numpy())].mean()
            hungarian_scores.append(h_score.item())
            d_matrix['YS'] = d_matrix['YS'][:orig_n_labels] # remove duplicates for GED computation

        for key in d_matrices.keys():
            d_matrices[key].append(d_matrix[key])

    for key in d_matrices.keys():
        d_matrices[key] = torch.stack(d_matrices[key], axis=0)

    ged, d_YS, d_SS = calc_energy_distances(d_matrices, flip_probs = probs)

    if return_hungarian:
        return ged, d_matrices, d_YS, d_SS, torch.tensor(hungarian_scores)
    else:
        return ged, d_matrices, d_YS, d_SS


@torch.no_grad()
def compute_stats(args, generator, images, calnet_preds, calnet_labelled_imgs, fake_labels, pred_dist, gan_al_maps, labels, gt_dist, gt_labels, ignore_mask, b_index=0): #todo log in tensorboardx

    ged = None

    if not args.calibration_net == "EmptyCalNet":

        calnet_iou = []
        for i in range(len(calnet_preds)):
            calnet_iou.append(compute_iou(calnet_preds[i], labels[i], args))

        if args.dataset == "LIDC":
            mean_calnet_iou = nanmean(torch.tensor(calnet_iou))
        else:
            mean_calnet_iou = nanmean(torch.stack(calnet_iou, dim=0))

        if not args.debug and args.mode == "train":
            wandb.log({"Calibration net predictions Mean IoU": mean_calnet_iou.cpu()})
    if not args.generator == "EmptyGenerator":
        gen_iou = []
        for i in range(len(fake_labels)):
            gen_iou.append(compute_iou(fake_labels[i], labels[i], args))

        if args.dataset == "LIDC":
            mean_gen_iou = nanmean(torch.tensor(gen_iou))
        else:
            mean_gen_iou = nanmean(torch.stack(gen_iou, dim=0))


        if not args.debug and args.mode == "train":
            wandb.log({"Final predictions Mean IoU": mean_gen_iou.cpu()})


    if args.dataset == "LIDC":
        if not args.generator == "EmptyGenerator":
            ged, _, d_YS, d_SS, h_scores = compute_ged(pred_dist, gt_dist, calnet_preds, args=args, n_samples=args.n_generator_samples_test, return_hungarian=True)

            ged = ged.mean()
            d_YS = d_YS.mean()
            d_SS = d_SS.mean()
            h_scores = nanmean(h_scores)

            if not args.debug and args.mode == "train":
                wandb.log({"meanGED": ged.cpu()})  # todo make sure GED is correct!
                wandb.log({"meanYS": d_YS.cpu()})
                wandb.log({"meanSS": d_SS.cpu()})
                if args.dataset == "LIDC": wandb.log({"hungarian": h_scores})

    if args.dataset == "CITYSCAPES19":

        # get classes, corresponding flip classes and probabilities
        flip_args = eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")
        class_1 = flip_args[0]
        class_2 = flip_args[1]
        flip_probs = flip_args[2]

        if args.generator == "EmptyGenerator":
            calnet_class_probs = compute_pred_class_probs(labels, calnet_preds, ignore_mask, args)
            gt_class_probs = compute_gt_class_probs(labels, args)
            with torch.no_grad():
                calnet_precision = torch.abs(calnet_class_probs - gt_class_probs).sum()

                f_classes = class_1 + class_2
                calnet_flip_precision = torch.abs(np.take(calnet_class_probs, f_classes) - np.take(gt_class_probs, f_classes)).sum()

        else:
            pred_class_probs = compute_pred_class_probs(labels, pred_dist.mean(0), ignore_mask, args)
            gt_class_probs = compute_gt_class_probs(labels, args)
            with torch.no_grad():
                gen_precision = torch.abs(pred_class_probs - gt_class_probs).sum()

                f_classes = class_1 + class_2
                gen_flip_precision = torch.abs(np.take(pred_class_probs, f_classes) - np.take(gt_class_probs,  f_classes)).sum()


        if not args.debug and args.mode == "train":
            if args.generator == "EmptyGenerator":
                wandb.log({"calnet_precision": calnet_precision.cpu()})  # todo make sure GED is correct!
                wandb.log({"calnet_flip_precision": calnet_flip_precision.cpu()})
            else:
                wandb.log({"gen_precision": gen_precision.cpu()})  # todo make sure GED is correct!
                wandb.log({"gen_flip_precision": gen_flip_precision.cpu()})

        if instance_checker(generator, GeneralVAE):

            label = gt_labels[:,1,:,:] # rgb channels are identical so we extract only one of them

            gt_dist = get_all_modes(label, input_classes = class_1, target_classes=class_2, flip_probs = flip_probs, n_flipped_modes = len(flip_probs))

            ged, _, d_YS, d_SS = compute_ged(pred_dist, gt_dist, calnet_preds, args=args, n_samples=args.n_generator_samples_test)

            ged = ged.mean()
            d_YS = d_YS.mean()
            d_SS = d_SS.mean()

            if not args.debug and args.mode == "train":
                wandb.log({"meanGED": ged.cpu()})  # todo make sure GED is correct!
                wandb.log({"meanYS": d_YS.cpu()})
                wandb.log({"meanSS": d_SS.cpu()})

    return ged


def get_all_modes(label, input_classes, target_classes, flip_probs, n_flipped_modes):
    mode_stats = get_mode_statistics(probabilities=flip_probs, n_flipped_modes= n_flipped_modes)
    flips = mode_stats["flips"]

    # get all modes
    gt_dist = torch.zeros(label.shape[0], len(flips), label.shape[1], label.shape[2])

    for m in range(len(flips)):
        gt_dist[:,m] = label
        for c in range(flips.shape[1]):
            if flips[m][c]:
                gt_dist[:,m][label == input_classes[c]] = target_classes[c]

    return gt_dist

def count_pixel_modes(labels, pred_dist, input_classes, target_classes):

    # convert from 1-hot to discrete
    labels = labels.argmax(1)
    pred_dist = pred_dist.argmax(2)

    pixel_mode_counts = torch.zeros(len(labels), len(input_classes), 3)

    for i, label in enumerate(labels):
        for j,c in enumerate(input_classes):
            # find indexes of pixels in that class in the gt label
            c_idxs = torch.where(label == c)

            total_c_pixels = torch.sum(label==c) *  len(pred_dist)
            pixel_mode_counts[i, j, 0] = total_c_pixels

            for k in range(len(pred_dist)):

                pred = pred_dist[k,i]

                # count how many pixels in the class mask belong to the original class or to the flipped class
                pred_class1_counts = torch.sum(pred[c_idxs]==input_classes[j])
                pred_class2_counts = torch.sum(pred[c_idxs]==target_classes[j])

                pixel_mode_counts[i, j, 1] += pred_class1_counts
                pixel_mode_counts[i, j, 2] += pred_class2_counts

    return pixel_mode_counts

def compute_gt_class_probs(labels, args):
    gt_class_probs = []

    # get idxs of all classes and of only switchable classes
    classes = eval(f"CITYSCAPES19_COLOUR_ENCODING")
    var_classes = eval(f"CITYSCAPES19_FLIP_CLASSES")

    flip_details = eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")
    f_classes = flip_details[1]

    for c_ind,c in enumerate(classes):
        if c_ind in f_classes:
            c_idx = flip_details[0][f_classes.index(c_ind)]
        else:
            c_idx = c_ind
        if c_idx in labels.argmax(1):
            if c in var_classes:
                gt_class_probs.append(1 - var_classes[c][1])
            elif c[:-1] in var_classes:
                gt_class_probs.append(var_classes[c[:-1]][1])
            else:
                gt_class_probs.append(1)
        else:
            gt_class_probs.append(0)

    return torch.tensor(gt_class_probs)

def compute_pred_class_probs(labels, preds, ignore_idxs, args):

    if ignore_idxs is None:
        ignore_idxs = torch.ones(labels.argmax(1).shape)

    pn_class_probs = []
    classes = eval(f"CITYSCAPES19_COLOUR_ENCODING")

    flip_details = eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")

    f_classes = flip_details[1]

    for c_ind, c in enumerate(classes):
        if c_ind in f_classes:
            c_idx = flip_details[0][f_classes.index(c_ind)]
        else:
            c_idx = c_ind

        # c_idxs = torch.where(labels.argmax(1) == c_idx) #todo add ign idxs here?

        c_idxs = torch.where((labels.argmax(1) == c_idx)*(ignore_idxs))

        pn_class_probs.append(torch.mean(preds[c_idxs[0], c_ind, c_idxs[1], c_idxs[2]]))

    ps =  torch.tensor(pn_class_probs)

    ps[torch.isnan(ps)]= 0

    return ps


def save_numpy_arrays(images, labels, calnet_preds, pred_dist, batch_id, args):

    saving_images = images.cpu().numpy()
    saving_labels = labels.cpu().numpy()
    saving_calnet_preds = calnet_preds.cpu().numpy()
    saving_pred_dist = pred_dist.detach().cpu().numpy()

    data = {"images": saving_images, "labels": saving_labels, "calnet_preds": saving_calnet_preds,
            "pred_dist": saving_pred_dist}

    if args.mode == "train":
        stamp = DATA_MANAGER.stamp
    elif args.mode == "test":
        stamp = args.test_model_date

    saving_path = Path(f'./{PREFIX_OUTPUT}/{stamp}/{NUMPY_DIR}/{args.mode}/')


    # if directory does not exist, make it
    saving_path.mkdir(parents=True, exist_ok=True)

    np.savez(str(saving_path) + f'/batch_{batch_id}.npz', **data)


def load_numpy_arrays(batch_id, args, to_tensor=False):
    """
    Load saved numpy arrays of images (B x C x H x W), labels (B x C x H x W),
    calibration net predictions (B x C x H x W) and predictive distribution (N x B x C x H x W)
    """

    if args.mode == "train":
        stamp = DATA_MANAGER.stamp
    elif args.mode == "test":
        stamp = args.test_model_date

    loading_path = Path(f'./{PREFIX_OUTPUT}/{stamp}/{NUMPY_DIR}/') # todo: replace this with your path

    # if directory does not exist, make it
    assert loading_path.exists()

    data = np.load(str(loading_path) + f"/{args.mode}/batch_{batch_id}.npz")

    images, labels, calnet_preds, pred_dist = data["images"], data["labels"], data["calnet_preds"], data["pred_dist"]

    if to_tensor:
        images = torch.from_numpy(images).float().to(DEVICE)
        labels = torch.from_numpy(labels).to(DEVICE)
        calnet_preds = torch.from_numpy(calnet_preds).float().to(DEVICE)
        pred_dist = torch.from_numpy(pred_dist).float().to(DEVICE)

    return images,labels,calnet_preds, pred_dist

def save_results(gt_mode_probs, flip_probs, calnet_class_probs, pred_class_probs, mGED, mYS, mSS, args):

    # plot figures
    flip_cal_plot = plot_calibration(flip_probs, calnet_class_probs, pred_class_probs)
    flip_cal_plot = torch.from_numpy(np.moveaxis(flip_cal_plot, -1, 0)).float()


    gt_mode_probs = gt_mode_probs.cpu().numpy()
    flip_probs = np.array(flip_probs)
    calnet_class_probs = calnet_class_probs.cpu().numpy()
    pred_class_probs = pred_class_probs.cpu().numpy()
    mGED = mGED.cpu().numpy()
    mYS = mYS.cpu().numpy()
    mSS = mSS.cpu().numpy()

    data = {"gt_mode_probs": gt_mode_probs, "flip_probs": flip_probs, "calnet_class_probs": calnet_class_probs, "pred_class_probs":pred_class_probs, "mGED":mGED, "mYS":mYS, "mSS":mSS}

    if args.mode == "train":
        stamp = DATA_MANAGER.stamp
    elif args.mode == "test":
        stamp = args.test_model_date

    saving_path = Path(f'./{PREFIX_OUTPUT}/{stamp}/{NUMPY_DIR}/{args.mode}/results/')

    # if directory does not exist, make it
    saving_path.mkdir(parents=True, exist_ok=True)

    np.savez(str(saving_path) + f'/results.npz', **data)

    # save plots
    plot_saving_path = Path(f'./{PREFIX_OUTPUT}/{stamp}/{NUMPY_DIR}/{args.mode}/results/plots/')

    # if directory does not exist, make it
    plot_saving_path.mkdir(parents=True, exist_ok=True)

    normalize = True if flip_cal_plot.max() > 1 and flip_cal_plot.min() >= 0 else False
    save_image(flip_cal_plot, f'{str(plot_saving_path)}/flip_cal_plot.png', nrow=4, normalize=normalize)

def load_results(args, to_tensor=False): #TODO ADD RUN NAME HERE

    if args.mode == "train":
        stamp = DATA_MANAGER.stamp
    elif args.mode == "test":
        stamp = args.test_model_date

    loading_path = Path(f'./{PREFIX_OUTPUT}/{stamp}/{NUMPY_DIR}/test/results') # todo: replace this with your path

    # if directory does not exist, make it
    assert loading_path.exists()

    data = np.load(str(loading_path) + f"/results.npz")

    gt_mode_probs, mean_mode_counts, flip_probs, calnet_class_probs, pred_class_probs, mGED, mYS, mSS  = data["gt_mode_probs"], data["mean_mode_counts"], data["flip_probs"], data["calnet_class_probs"], data["pred_class_probs"], data["mGED"], data["mYS"], data["mSS"]

    if to_tensor:
        gt_mode_probs = torch.from_numpy(gt_mode_probs).float().to(DEVICE)
        mean_mode_counts = torch.from_numpy(mean_mode_counts).float().to(DEVICE)
        flip_probs = list(flip_probs)
        calnet_class_probs = torch.from_numpy(calnet_class_probs).float().to(DEVICE)
        pred_class_probs = torch.from_numpy(pred_class_probs).float().to(DEVICE)
        mGED = torch.from_numpy(mGED).float().to(DEVICE)
        mYS = torch.from_numpy(mYS).float().to(DEVICE)
        mSS = torch.from_numpy(mSS).float().to(DEVICE)

    return gt_mode_probs, mean_mode_counts, flip_probs, calnet_class_probs, pred_class_probs, mGED, mYS, mSS

def plot_comparison_figure(batch, calnet_preds, fake_labels, al_maps, gan_al_maps, generator, calibration_net, discriminator, args):

    if args.dataset == "LIDC":
        images, labels, gt_dist = unpack_batch(batch)
        gt_labels = None
        lidc_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    else:
        images, labels = unpack_batch(batch)
        gt_dist = None
        gt_labels = None
        lidc_norm = None

    if (args.dataset == "CITYSCAPES19" and args.class_flip):
        gt_labels = labels.clone()
        labels = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2)
        bb_preds = batch["bb_preds"].to(DEVICE).float()
        bb_preds = torch.eye(LABELS_CHANNELS)[bb_preds[:, 1, :, :].long()].permute(0, 3, 1, 2).to(DEVICE)
    else:
        bb_preds = None


    # check used model types
    use_calnet = args.calibration_net != "EmptyCalNet"
    use_generator = args.generator != "EmptyGenerator"

    # free up some space
    del (calibration_net, generator, discriminator)

    # initialize figure size arguments
    n_pics = 5
    n_plots = 4 if not use_calnet else 6
    n_plots = n_plots + 1 if args.dataset == "CITYSCAPES19" else n_plots

    # initialize figure
    fig = plt.figure(figsize=(n_plots * 2 + 2, n_pics * 2))
    canvas = FigureCanvasAgg(fig)

    for idx in range(n_pics):

        extra = 0

        # convert to plottable format
        plottable_images = move_color_channel(de_torch((images[idx] + 1) / 2))  # +1/2 to normalize between 0 and 1
        if plottable_images.shape[-1]==1: plottable_images = plottable_images.squeeze()

        if args.dataset == "LIDC":
            pad = lambda x: np.pad(x.cpu().numpy(), pad_width=2, mode='constant', constant_values=1)
            glued_top = np.concatenate((pad(gt_dist[idx, 0]), pad(gt_dist[idx, 1])), axis=1)
            glued_bottom = np.concatenate((pad(gt_dist[idx, 2]), pad(gt_dist[idx, 3])), axis=1)
            plottable_t_labels = np.concatenate([glued_top, glued_bottom], axis=0)
        else:
            plottable_t_labels = _recolour_label(de_torch(_1hot_2_2d(labels[idx],sample=True)), dataset=args.dataset)

        if args.dataset == "CITYSCAPES19":
            plottable_bb_preds = _recolour_label(de_torch(_1hot_2_2d(bb_preds[idx],sample=True)), dataset=args.dataset)

        if use_generator:
            plottable_f_labels = _recolour_label(de_torch(_1hot_2_2d(fake_labels[idx],sample=True)), dataset=args.dataset)

        if use_calnet:
            plottable_al_maps = de_torch(al_maps[idx])
            plottable_calnet_preds = _recolour_label(de_torch(_1hot_2_2d(calnet_preds[idx], sample=True)), dataset=args.dataset)

        if use_generator:
            plottable_gan_al_maps = de_torch(gan_al_maps[idx])

        # plot figure

        # input image
        plt.subplot(n_pics, n_plots, idx * n_plots + 1)
        plt.imshow(plottable_images, interpolation="none")
        if idx == 0: plt.title("Input Image")
        plt.xticks([])
        plt.yticks([])

        # true label
        plt.subplot(n_pics, n_plots, idx * n_plots + 2)
        plt.imshow(plottable_t_labels, norm=lidc_norm, interpolation="none")
        if idx == 0: plt.title("Label")
        plt.xticks([])
        plt.yticks([])

        if args.dataset == "CITYSCAPES19":
            # black-box net prediction
            extra += 1
            plt.subplot(n_pics, n_plots, idx * n_plots + 2 + extra)
            plt.imshow(plottable_bb_preds, interpolation="none")
            if idx == 0: plt.title("BB Pred")
            plt.xticks([])
            plt.yticks([])


        if use_calnet:
            # calibration net prediction
            plt.subplot(n_pics, n_plots, idx * n_plots + 3 + extra)
            plt.imshow(plottable_calnet_preds,norm=lidc_norm, interpolation="none")
            if idx == 0: plt.title("CalNet Pred")
            plt.xticks([])
            plt.yticks([])

            extra += 1

        if use_generator:
            # final prediction
            plt.subplot(n_pics, n_plots, idx * n_plots + 3 + extra)
            plt.imshow(plottable_f_labels, norm=lidc_norm, interpolation="none")
            if idx == 0: plt.title("RefNet Pred")
            plt.xticks([])
            plt.yticks([])

            extra += 1

        if use_calnet:
            # calibration pred aleatoric uncertainty
            plt.subplot(n_pics, n_plots, idx * n_plots + 3 + extra)

            al_norm = matplotlib.colors.Normalize(vmin=0, vmax=MAX_ALEATORIC)  # set range into which we normalize the aleatoric unc maps

            # make sure the aleatoric uncertainty is within range
            assert al_maps.max() <= MAX_ALEATORIC_GT, "Predicted aleatoric uncertainty not within range: True = 0 < " + str(
                    MAX_ALEATORIC) + ", Plottable = " + str(al_maps.min().item()) + " < " + str(al_maps.max().item())

            plt.imshow(plottable_al_maps, cmap='hot', norm=al_norm, interpolation="none")
            if idx == 0: plt.title("CalNet Aleatoric")
            plt.xticks([])
            plt.yticks([])
            extra += 1

        if use_generator:
            # generator aleatoric uncertinty
            plt.subplot(n_pics, n_plots, idx * n_plots + 3 + extra)
            al_norm = matplotlib.colors.Normalize(vmin=0, vmax=MAX_ALEATORIC)  # set range into which we normalize the aleatoric unc maps
            plt.imshow(plottable_gan_al_maps, cmap='hot', norm=al_norm, interpolation="none")
            if idx == 0: plt.title("RefNet Aleatoric")
            plt.xticks([])
            plt.yticks([])
            extra += 1

    canvas.draw()
    _, (width, height) = canvas.print_to_buffer()
    s = canvas.tostring_rgb()

    plt.close(fig)
    return np.fromstring(s, dtype='uint8').reshape((height, width, 3))


def plot_calibration_figure(labels, calnet_preds, pred_dist, ignore_idxs, args): #TODO DEVELOPING

    f_classes_idxs = torch.LongTensor(eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")[1])
    calnet_class_probs=compute_pred_class_probs(labels, calnet_preds, ignore_idxs, args)[f_classes_idxs]
    gt_class_probs = compute_gt_class_probs(labels, args)[f_classes_idxs]

    if pred_dist is not None:
        preds_class_probs= compute_pred_class_probs(labels, pred_dist.mean(0), ignore_idxs, args)[f_classes_idxs]

    fig, ax = plt.subplots()  # plt.figure(figsize=(5,5))
    canvas = FigureCanvasAgg(fig)

    labels = ['sidewalk', 'person', 'car', 'vegetation', 'road']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    rects1 = ax.bar(x - width, gt_class_probs, width, label='gt')
    rects2 = ax.bar(x, calnet_class_probs, width, label='calnet')
    if pred_dist is not None:
        rects3 = ax.bar(x + width, preds_class_probs, width, label='final')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Flip Probability')
    ax.set_title('Sample frequency vs. gt flip probability')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # plt.show()
    # error

    canvas.draw()
    _, (width, height) = canvas.print_to_buffer()
    s = canvas.tostring_rgb()
    plt.close(fig)
    return np.fromstring(s, dtype='uint8').reshape((height, width, 3))

def plot_sample_preds(images, labels, calnet_preds, pred_dist, dataset):

    n_plotted_preds = 5

    n_cols = n_plotted_preds + 4
    n_rows = pred_dist.shape[1]

    fig = plt.figure(figsize=(n_cols + 2, n_rows + 2))
    canvas = FigureCanvasAgg(fig)

    # plot sample predictions
    for i in range(n_rows):

        # plot input
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        plottable_images = move_color_channel(de_torch((images[i] + 1) / 2))
        if plottable_images.shape[-1] == 1: plottable_images = plottable_images.squeeze()
        plt.imshow(plottable_images, interpolation="none")
        if i == 0: plt.title("Input")
        plt.xticks([])
        plt.yticks([])

        for j in range(n_cols - 4):
            plottable_pred = _recolour_label(de_torch(_1hot_2_2d(pred_dist[j, i, :, :], sample=True)), dataset=dataset)
            plt.subplot(n_rows, n_cols, i * n_cols + j + 2)
            plt.imshow(plottable_pred, interpolation="none")
            if i == 0: plt.title(f"Pred {j + 1}")
            plt.xticks([])
            plt.yticks([])

        # plot average predictions
        plt.subplot(n_rows, n_cols, i * n_cols + n_cols - 2)
        plottable_avg_pred = _recolour_label(de_torch(_1hot_2_2d(pred_dist[:, i, :, :].mean(0), sample=True)),
                                             dataset=dataset)
        plt.imshow(plottable_avg_pred, interpolation="none")
        if i == 0: plt.title(f"Avg Pred\nN = {pred_dist.shape[0]}")
        plt.xticks([])
        plt.yticks([])

        # plot calibration net predictions
        plt.subplot(n_rows, n_cols, i * n_cols + n_cols - 1)
        plottable_calnet_pred = _recolour_label(de_torch(_1hot_2_2d(calnet_preds[i], sample=True)), dataset=dataset)
        plt.imshow(plottable_calnet_pred, interpolation="none")
        if i == 0: plt.title("Cal Pred")
        plt.xticks([])
        plt.yticks([])

        # plot actual predictions
        plt.subplot(n_rows, n_cols, i * n_cols + n_cols)

        if labels.shape[1] != LABELS_CHANNELS:
            label = torch.eye(LABELS_CHANNELS)[labels[:, 1, :, :].long()].permute(0, 3, 1, 2)[
                i]  # convert rgb label to one-hot
        else:
            label = labels[i]

        plottable_label = _recolour_label(de_torch(_1hot_2_2d(label, sample=True)), dataset=dataset)
        plt.imshow(plottable_label, interpolation="none")
        if i == 0: plt.title("Label 0")
        plt.xticks([])
        plt.yticks([])

    fig.suptitle('Sample predictions')

    # convert figure to array
    canvas.draw()
    _, (width, height) = canvas.print_to_buffer()
    s = canvas.tostring_rgb()

    plt.close(fig)

    return np.fromstring(s, dtype='uint8').reshape((height, width, 3))

def plot_calibration(gt_class_probs, calnet_cprobs, pred_cprobs):
    calnet_cprobs = list(calnet_cprobs[:, 1].cpu().numpy())
    pred_cprobs = list(pred_cprobs[:, 1].cpu().numpy())

    fig, ax = plt.subplots()  # plt.figure(figsize=(5,5))
    canvas = FigureCanvasAgg(fig)

    labels = ['sidewalk', 'person', 'car', 'vegetation', 'road']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    rects1 = ax.bar(x - width, gt_class_probs, width, label='gt')
    rects2 = ax.bar(x, calnet_cprobs, width, label='calnet')
    rects3 = ax.bar(x + width, pred_cprobs, width, label='final')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Flip Probability')
    ax.set_title('Sample frequency vs. gt flip probability')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    canvas.draw()
    _, (width, height) = canvas.print_to_buffer()
    s = canvas.tostring_rgb()
    plt.close(fig)

    return np.fromstring(s, dtype='uint8').reshape((height, width, 3))



