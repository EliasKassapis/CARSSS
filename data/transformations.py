from typing import Callable, List, Optional

import cv2
from torchvision import transforms

from utils import constants, general_utils
from utils.constants import *
from utils.training_helpers import *
from numpy import random


def _apply_to_all(sample, sample_setup, fn):

    process_sample, *args = sample_setup(sample)

    for key, value in sample.items():
        sample[key] = fn(value, *args)

    return sample


def _process_all(sample):
    return True, None


class RandomHorizontalFlip:
    """RandomHorizontalFlip should be applied to all n images together, not just one

    """
    def __init__(self, probability: float = 0.5):
        self._probability = probability

    def __call__(self, sample):
        if random.random() < self._probability:
            return _apply_to_all(sample, _process_all, self._fn)
        else:
            return sample

    @staticmethod
    def _fn(value: np.ndarray, *args):
            return value

class ClassFlip:
    """Random Class Flip should be applied to all n images together, not just one
    """
    def __init__(self, c1 = [3, 9, 8, 5, 4], c2 = [13, 12, 14, 15, 16], p=[8 / 17, 7 / 17, 6 / 17, 5 / 17, 4 / 17]):

        assert len(c1) == len(c2) == len(p), f"Missmatch between number of labels to be switched and probs. l1 = {len(c1)}, l2 = {len(c2)} and p = {len(p)}"

        self.labels1 = c1
        self.labels2 = c2
        self.probabilities = p

    def __call__(self, sample):
        return _apply_to_all(sample, self.pass_args, self._fn)

    def pass_args(self, sample):

        return (
            True,
            self.labels1,
            self.labels2,
            self.probabilities
        )

    @staticmethod
    def _fn(value, *args):

        labels1, labels2, probability = args

        if value.max() <= len(CITYSCAPES19_N_LABELS): # perform only on labels

            if value.shape[-1] == 3:

                roll = np.random.rand(len(probability))
                flip_idxs = roll < probability

                for i in range(len(probability)):
                    if flip_idxs[i]:
                        value[value==labels1[i]]=labels2[i]

            else: # if 1st dim is batch dim

                roll = np.random.rand(len(value), len(probability))
                flip_idxs = roll < probability

                for j in range(len(value)):
                    for i in range(len(probability)):
                        if flip_idxs[j][i]:
                            value[j][value[j]==labels1[i]]=labels2[i]

        return value


class RandomRescale:
    def __init__(self, probability: float = 1 / 3, scales = None):
        self._probability = probability
        if scales is None:
            scales = [1.1, 1.2]
        self._scales = scales

    def __call__(self, sample):
        return _apply_to_all(sample, self._process_s, self._fn)

    def _process_s(self, sample):
        return random.random() < self._probability, random.choice(self._scales)

    @staticmethod
    def _fn(value, *args):
        scale = args[0]

        return cv2.resize(
            value, None, fx=scale, fy=scale, interpolation=constants.INTERPOLATION
        )


class RandomScaleCrop:
    def __init__(
        self, imsize, probability = 1 / 2, scales = None
    ):
        self._probability = probability
        if scales is None:
            scales = [0.8, 0.9, 0.95]
        self._scales = scales
        self.imsize = imsize
        self.shear_factor = np.concatenate((np.linspace(0.05,0.1,4), np.linspace(-0.1,-0.05,4)),axis=0)

    def __call__(self, sample):
        return _apply_to_all(sample, self._process_s, self._fn)

    def _process_s(self, s: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        scale = random.choice(self._scales)

        shear_f = random.choice(self.shear_factor) if random.random() < self._probability else 0

        input_height, input_width, _ = s['image'].shape

        if self.imsize[0] == self.imsize[1]:
            t_size = int(input_height * scale) if input_height < input_width else int(input_width * (scale + shear_f) )
            t_size = (t_size, t_size)
        else:
            t_size = (int(input_height * scale), int(input_width * scale))


        target_height, target_width = (
            t_size[0],
            t_size[1],
        )
        top = np.random.randint(0, input_height - target_height)
        left = np.random.randint(0, input_width - target_width)
        return (
            random.random() < self._probability,
            target_height,
            target_width,
            top,
            left,
        )

    @staticmethod
    def _fn(value, *args):
        target_height, target_width, top, left = args
        out =  value[top : top + target_height, left : left + target_width]

        return out

class RCrop:
    def __init__(
        self, probability = 1 / 3, imsize = None
    ):
        self._probability = probability

        self._imsize = imsize

    def __call__(self, sample):
        return _apply_to_all(sample, self._process_s, self._fn)

    def _process_s(self, sample):
        input_height, input_width, _ = sample['image'].shape

        t_size = self._imsize

        target_height, target_width = (
            t_size[0],
            t_size[1],
        )

        top = np.random.randint(0, input_height - target_height) if input_height != target_height else 0
        left = np.random.randint(0, input_width - target_width) if input_width != target_width else 0
        return (
            random.random() < self._probability,
            target_height,
            target_width,
            top,
            left,
        )

    @staticmethod
    def _fn(value: np.ndarray, *args) -> np.ndarray:
        target_height, target_width, top, left = args
        return value[top : top + target_height, left : left + target_width]

class Crop:
    def __init__(self, imsize = None):
        self._imsize = imsize

    def __call__(self, sample):
        return _apply_to_all(sample, self._process_s, self._fn)

    def _process_s(self, sample):
        input_height, input_width, _ = sample['image'].shape
        t_size = self._imsize

        target_height, target_width = (
            t_size[0],
            t_size[1],
        )

        # get a center cropping
        top = (input_height - target_height)//2
        left = (input_width - target_width)//2

        return (
            True,
            target_height,
            target_width,
            top,
            left,
        )

    @staticmethod
    def _fn(value, *args):
        target_height, target_width, top, left = args
        if value.shape[0] == 4:
            out = value[:, top: top + target_height, left: left + target_width]
        else:
            out = value[top: top + target_height, left: left + target_width]

        return out

class Resize:

    def __call__(self, sample):
        return _apply_to_all(sample, _process_all, self._fn)

    @staticmethod
    def _fn(value, *args):
        reshaped = False

        try:
            if value.shape == (4,180,180):
                reshaped = True
                value = np.moveaxis(value, 0,-1)
            width, height, n_channels = value.shape
        except Exception:
            width, height = value.shape
            n_channels = 0

        if width == constants.IMSIZE[0] and height == constants.IMSIZE[1]:
            return value
        elif n_channels <= 3:

            return cv2.resize(
                value,
                (constants.IMSIZE[1], constants.IMSIZE[0]),
                interpolation=constants.INTERPOLATION,
            )
        else:
            channel_list = []
            channels = cv2.split(value)
            for channel in channels:
                new_channel = cv2.resize(
                    channel,
                    (constants.IMSIZE[1], constants.IMSIZE[0]),
                    interpolation=constants.INTERPOLATION,
                )
                channel_list.append(new_channel)
            out = cv2.merge(channel_list)
            if reshaped:
                out = np.moveaxis(out, -1, 0)
            return out


class RescaleValues:
    def __call__(self, sample):
        return _apply_to_all(sample, _process_all, self._fn)

    @staticmethod
    def _fn(value, *args):

        if value.max() > LABELS_CHANNELS and len(value.shape) == 3: # if input is an RGB image

            value = value.astype(float)
            value = (value / 255) * 2 - 1

            assert -1 <= value.min() <= value.max() <= 1

        else:
            return value.astype(float) # do not rescale labels

        return value

class _2d_to_1hot:
    def __call__(self, sample):
        return _apply_to_all(sample, _process_all, self._fn)

    @staticmethod
    def _fn(value, *args):
        # if input is an rgb image or unc map
        if (value.max() > LABELS_CHANNELS and len(value.shape)== 3) or value.dtype != np.uint8 or value.shape == (4,128,128):
            return value
        else:
            assert value.shape[2] == 3 or value.shape[2] == 2, f"Unexpected input to convert to 1-hot format. Expected RGB or grayscale image, got input of shape = {value.shape}"

            labels = value[:, :, 1]

            if labels.dtype != np.uint8:
                print(f"Label not uint8! Shape = {labels.shape}, value shape = {value.shape}, minmax = {value.min()} < {value.max()}")
                try:
                    labels = labels.astype(np.uint8)
                except Exception as e:
                    print(e, f" labels dtype = {labels.dtype}")

            one_hot = np.eye(LABELS_CHANNELS)[labels] #todo check if -1 is correct here!!!!

            return one_hot

class ChangeChannels:
    def __call__(self, sample):
        return _apply_to_all(sample, _process_all, self._fn)

    @staticmethod
    def _fn(value, *args):
        if value.shape != (4,128,128): #do not move channel of TOY gt dristribution!
            value = np.moveaxis(value, -1, 0)
        return value
