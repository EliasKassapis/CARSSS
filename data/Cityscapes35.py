import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils.constants import DEVICE

import cv2
from data import transformations
from utils import personal_constants
from utils.data_utils import _recolour_label, _1hot_2_2d
from torchvision.datasets import Cityscapes

class Cityscapes35(Cityscapes): #todo place limit on dataset!!!!!

    def __init__(
            self,
            mode,
            transform = None,
    ):

        super(Cityscapes35, self).__init__(target_type="semantic",mode="fine", split= mode.lower(), transform=transform, target_transform=transform,root=personal_constants.CITYSCAPES_PATH)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = self._load_image(self.images[index])

        targets = []

        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                raise Exception
                target = self._load_json(self.targets[index][i])
            else:
                target = self._load_image(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        sample = {
            'image': image,
            'label': target
        }

        if self.transform:
            sample = self.transform(sample)

        return (sample)


    def _load_image(self, frame_input_path: Path) -> np.ndarray:

        image = cv2.imread(frame_input_path)

        return image


if __name__ == '__main__':

    from torchvision import transforms
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    from utils.training_helpers import unpack_batch
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transformations.RandomHorizontalFlip(),
            transformations.RandomScaleCrop(probability=1, scales=[0.6, 0.7, 0.8]),
            transformations.Resize(),
            transformations._2d_to_1hot(),
            transformations.RescaleValues(),
            transformations.ChangeChannels(),
        ]
    )

    dataset = Cityscapes35(mode="train", transform=transform)

    batch_size = 3

    data = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=4)

    data_bar = tqdm(data)

    for i, (batches) in enumerate(data_bar):
        # Visualize batches for DEBUG

        batch_1 = batches

        image_1, labels_1 = unpack_batch(batch_1)

        labels_1 = _recolour_label(_1hot_2_2d(labels_1,dim=1), dataset="CITYSCAPES35").permute(0,3,1,2).float().to(DEVICE)

        batch = torch.cat((image_1, labels_1), dim=0)

        plt.figure(1)
        plt.imshow(vutils.make_grid(batch, nrow=batch_size, normalize=True).cpu().numpy().transpose(1, 2, 0))
        plt.show()
