import torch
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import Dataset

import cv2
from data import transformations
from utils import personal_constants, constants
from tqdm import tqdm
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils.data_utils import _1hot_2_2d
from utils.training_helpers import unpack_batch


class LIDC(Dataset):

    def __init__(
            self,
            mode,
            transform= None,
    ):

        self.mode = mode

        # data_path = personal_constants.LIDC_RAW_PATH / mode.lower() / f'images'
        # labels_path = personal_constants.LIDC_RAW_PATH / mode.lower() / f'gt'

        data_path = personal_constants.LIDC_RAW_PATH / f"lidc_crops_{mode.lower()}" / mode.lower() / f'images'
        labels_path = personal_constants.LIDC_RAW_PATH / f"lidc_crops_{mode.lower()}" / mode.lower() / f'gt'

        assert mode in ["train", "test","val"], 'Incorrect dataset mode. Accepted modes include: "train", "test" or "val"'

        self._all_imgs = self.all_file_paths(data_path)
        self._all_labels = self.all_file_paths(labels_path)

        self._n_images = len(self._all_imgs)
        print(f'n images in {mode} dataset: {self._n_images}')

        self._transform = transform

    def _sum(self, video_path):

        sum_imgs = len(list((video_path.iterdir())))

        return sum_imgs

    def __len__(self):
        return self._n_images

    def __getitem__(self, index):

        # get image
        image = self._load_image(self._all_imgs[index])

        # get index of first label out of four
        get_label_idx = lambda x: (x + 1) * 4 - 4
        label0_idx = get_label_idx(index)

        # get all possible labels for source image
        label_idxs = list(range(label0_idx, label0_idx+4))

        assert self._all_labels[label0_idx].stem.split("_")[0] == self._all_imgs[index].stem.split("_")[0], f"Label is for patient {self._all_labels[label0_idx].stem.split('_')[0]} whereas Image is for patient {self._all_imgs[index].stem.split('_')[0]}"
        assert len(set([self._all_labels[i].stem.split("_")[0] for i in label_idxs]))==1

        labels = np.stack([self._load_image(self._all_labels[idx]) for idx in label_idxs],axis=0)
        labels[labels==255] = 1 # values are 0 and 255 in loaded labels

        # sample random label
        label_index = np.random.choice([0,1,2,3])
        label = labels[label_index]

        if self.mode == "train":
            sample = {
                'image': image,
                'label': label,
            }
        else:
            sample = {
                'image': image,
                'label': label,
                'dist': labels[:,:,:,0] # extract channel 0 because channel dims are identical
            }

        if self._transform:
            sample = self._transform(sample)

        return (sample)

    def all_file_paths(self, path):
        all_files = []
        for patient in path.iterdir():
            for p in patient.iterdir():
                all_files.append(p)

        all_files = sorted(all_files)

        return all_files


    def _load_image(self, frame_input_path):

        if not frame_input_path.exists():
            raise Exception(f'Image does not exist: {frame_input_path}')

        image = cv2.imread(str(frame_input_path))

        return image


if __name__ == '__main__':


    transform = transforms.Compose(
        [
            transformations.Crop(imsize=(128,128)),
            transformations._2d_to_1hot(),
            transformations.RescaleValues(),
            transformations.ChangeChannels(),
        ]
    )

    dataset = LIDC(mode="test", transform=transform)

    batch_size = 5

    data = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=0)

    plotting_batches = next(iter(data))
    batch_1 = plotting_batches

    data_bar = tqdm(data)

    for i, (batches) in enumerate(data_bar):
        # Visualize batches for DEBUG
        image_1, labels_1, dist = unpack_batch(batches)

        image_1 = (image_1+1)/2

        labels_1 = _1hot_2_2d(labels_1, dim=1).float().to(constants.DEVICE).unsqueeze(dim=1).repeat(1, 3, 1, 1)

        pad = lambda x: np.pad(x.cpu().numpy(), pad_width=2, mode='constant', constant_values=1)

        glued_top = np.concatenate((pad(dist[1,0]), pad(dist[1,1])), axis=1)
        glued_bottom = np.concatenate((pad(dist[1,2]),pad(dist[1,3])), axis=1)
        glued_all = np.concatenate([glued_top, glued_bottom], axis=0)

        batch = torch.cat((image_1, labels_1), dim=0)

        plt.figure(1)
        plt.imshow(vutils.make_grid(batch, nrow=batch_size, normalize=True).cpu().numpy().transpose(1, 2, 0))
        plt.show()

        plt.figure(2)

        plt.imshow(glued_all)

        plt.show()
