import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.constants import DEVICE, CITYSCAPES19_DEEPFLIP

from data import transformations
from utils import constants, personal_constants
from utils.data_utils import _recolour_label, _1hot_2_2d


class Cityscapes19(Dataset):

    def __init__(
            self,
            mode,
            transform = None,
    ):

            self.mode = mode

            data_path = personal_constants.CITYSCAPES_PATH / "processed" / "quarter" / mode.lower()

            assert mode in ["train", "test",
                            "val"], 'Incorrect dataset mode. Accepted modes include: "train", "test" or "val"'

            self._all_imgs = self.all_file_paths(data_path, type="image")
            self._all_labels = self.all_file_paths(data_path, type="label")
            self._all_prior_preds = self.all_file_paths(data_path, type= "prior")

            self._n_images = len(self._all_imgs)
            print(f'n images in {mode} dataset: {self._n_images}')

            self._transform = transform

    def _sum(self, video_path):

        sum_imgs = len(list((video_path.iterdir())))

        return sum_imgs

    def __len__(self):
        return self._n_images

    def __getitem__(self, index):

        assert str(self._all_imgs[index]).split("_leftImg8bit")[0] == str(self._all_labels[index]).split("_gtFine_trainIds")[0], "Images and Labels are not aligned"

        image = np.moveaxis(self._load_array(self._all_imgs[index]), 0, -1)

        label = self._load_array(self._all_labels[index])
        label[label == 255] = 24
        label = np.repeat(a=label[:,:, np.newaxis],repeats=[3], axis=2)
        unlabelled_idxs = np.where(label==24)

        prior = self._load_array(self._all_prior_preds[index])
        prior = np.repeat(a=prior[:, :, np.newaxis], repeats=[3], axis=2)
        masked_prior = self.encode_prior(prior)
        masked_prior[unlabelled_idxs] = 24 # classify unlabelled pixels as such

        sample = {
            'image': image,
            'label': label,
            'bb_preds': masked_prior
        }

        if self._transform:
            sample = self._transform(sample)


        assert sample['image'].shape[0] == 3, f"\nImage shape post transform = {sample['image'].shape}. Image shape pre-transform = {image.shape}. Image path = {self._all_imgs[index]} \nLabel shape post transform = {sample['label'].shape}. Label shape pre-transform = {label.shape}. Label path = {self._all_labels[index]}"
        if self.mode == "train":
            assert sample['label'].shape[0] == 25, f"\nLabel shape post transform = {sample['label'].shape}. Label shape pre-transform = {label.shape}. Label path = {self._all_labels[index]} \nImage shape post transform = {sample['image'].shape}. Image shape pre-transform = {image.shape}. Image path = {self._all_imgs[index]}"


        return (sample)


    def all_file_paths(self, path, type):
        all_files = []
        if type == "image":
            s = "leftImg8bit"
        elif type == "label":
            s = "gtFine_trainIds"
        elif type == "prior":
            s = "prior_preds_trainIds"

        for city in path.iterdir():
            if type == "prior":
                city = city / "prior_preds"
            for p in city.iterdir():
                if s in str(p.stem):
                    all_files.append(p)

        all_files = sorted(all_files)

        return all_files


    def _load_array(self, frame_input_path):

        if not frame_input_path.exists():
            raise Exception(f'Image does not exist: {frame_input_path}')

        array = np.load(str(frame_input_path))

        return array

    def encode_prior(self, prior):

        orig_labels = np.arange(-1,34)
        train_labels = np.array([24,24,24,24,24,24,24,24,0,1,24,24,2,3,4,24,24,24,5,24,6,7,8,9,10,11,12,13,14,15,24,24,16,17,18])

        arr = np.empty(orig_labels.max() + 1, dtype=np.uint8)
        arr[orig_labels] = train_labels
        masked_prior = arr[prior]

        return masked_prior

if __name__ == '__main__':

    from torchvision import transforms
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    from utils.training_helpers import unpack_batch
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transformations.Crop(imsize=constants.IMSIZE),
            transformations.RandomHorizontalFlip(),
            transformations.ClassFlip(*CITYSCAPES19_DEEPFLIP),
            transformations._2d_to_1hot(),
            transformations.RescaleValues(),
            transformations.ChangeChannels()
        ]
    )

    dataset = Cityscapes19(mode="train", transform=transform) #TODO NO TEST DIRECTORY IN PROCESSED

    batch_size = 8

    data = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=16)

    data_bar = tqdm(data)

    for i, (batches) in enumerate(data_bar):

        # Visualize batches for DEBUG
        batch_1 = batches

        image_1, labels_1 = unpack_batch(batch_1)

        p_preds = batch_1["bb_preds"].to(DEVICE).float()

        image_1 = 255*(image_1+1)/2
        labels_1 = _recolour_label(_1hot_2_2d(labels_1,dim=1), dataset="CITYSCAPES19").permute(0,3,1,2).float().to(DEVICE)
        p_preds = _recolour_label(_1hot_2_2d(p_preds, dim=1), dataset="CITYSCAPES19").permute(0, 3, 1, 2).float().to(DEVICE)

        batch = torch.cat((image_1, labels_1, p_preds), dim=0)

        plt.figure(figsize=(5,10))
        plt.imshow(vutils.make_grid(batch, nrow=batch_size, normalize=True).cpu().numpy().transpose(1, 2, 0))
        plt.show()