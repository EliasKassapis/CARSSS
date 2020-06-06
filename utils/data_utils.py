from utils import constants
from data import transformations
from utils.constants import *

def denormalize_picture(image, binarised = False):
    image = ((image + 1) / 2) * 255
    image[image > 255] = 255

    if (binarised):
        image[image == 127] = 255

    return image


def de_torch(img):
    return img.detach().cpu().numpy()


def move_color_channel(image):

    try:
        out = np.moveaxis(image, 0, -1)
        return out
    except Exception as e:
        print(f"Got {e}, input image shape = {image.shape}")


def BGR2RGB_numpy(image):
    temp = np.empty_like(image)
    temp[:, :, 0] = image[:, :, 2]
    temp[:, :, 1] = image[:, :, 1]
    temp[:, :, 2] = image[:, :, 0]
    return temp


def BGR2RGB_pytorch(images):
    temp = torch.empty_like(images)
    temp[:, 0, :, :] = images[:, 2, :, :]
    temp[:, 1, :, :] = images[:, 1, :, :]
    temp[:, 2, :, :] = images[:, 0, :, :]
    return temp


def _recolour_label(label, dataset, axis=0):

    if dataset == "LIDC":
        return label

    elif label.max() > 1:

        if type(label) != np.ndarray:
            label = label.cpu()

        colour_palette = eval(f"constants.{dataset}_NEW_COLOURS")

        label = np.take(colour_palette, label, axis=axis)

        return torch.from_numpy(label)
    else:
        return label


def _1hot_2_2d(label, sample=False, dim=0):

    if label.shape[2] == 3: #if the label is an rgb image normalize instead of discretizing
        return (label + 1)/2
    else:

        assert label.shape[2] != 3 or label.shape[
            3] != 2, "Unexpected input to convert to 2d format. Expected one-hot vector segmentation map"

        if constants.LABELS_CHANNELS == 2:

            dense = label.index_select(dim, torch.tensor(1).to(DEVICE)).squeeze()

        else:
            if sample:
                categorical = torch.distributions.one_hot_categorical.OneHotCategorical(probs=label.permute(1,2,0)) # move channel dim to last dim
                dense = categorical.sample().argmax(-1)
            else:
                dense = label.argmax(dim) # channel dimension = 0
        return dense

def get_transforms_list(mode, args):
    # add desired transformations to list
    transforms_list = []

    if args.dataset == "CITYSCAPES19":
        # crop 256x512 images to 128x128
        if mode == "train":
            if args.imsize != (256, 512):
                if args.imsize == (128, 128):
                    transforms_list += [
                        transformations.RandomScaleCrop(imsize=IMSIZE, probability=1, scales=[0.85])]
                    transforms_list += [transformations.RandomRescale(scales=np.linspace(.6, 1, 4))]
                transforms_list += [transformations.RCrop(imsize=IMSIZE)]
            else:
                transforms_list += [transformations.RandomScaleCrop(imsize=IMSIZE, scales=np.linspace(.6, .99, 4))]
                transforms_list += [transformations.Resize()]
        else:
            if args.imsize != (256, 512):
                transforms_list += [transformations.Crop(imsize=IMSIZE)]  # takes a central crop
    elif args.dataset == "LIDC":
        transforms_list += [transformations.Crop(imsize=IMSIZE)]
    else:
        if args.crop and mode == "train": transforms_list += [
            transformations.RandomScaleCrop(probability=1, scales=[0.6, 0.7, 0.8])]
        if args.resize: transforms_list += [transformations.Resize()]

    if not (args.dataset == "LIDC" and mode != "train"):
        if mode == "train": transforms_list += [transformations.RandomHorizontalFlip()]

    if (args.dataset == "CITYSCAPES19" and args.class_flip and mode == "train"):
        flip_args = eval(f"CITYSCAPES19_{args.flip_experiment}FLIP")
        transforms_list += [transformations.ClassFlip(*flip_args)]  # only flip in train mode!
    if not (args.dataset == "CITYSCAPES19" and args.class_flip and mode != "train"): # dont transform on CS val or test dataset
        transforms_list += [transformations._2d_to_1hot()]

    transforms_list += [transformations.RescaleValues(), transformations.ChangeChannels()]

    return transforms_list
