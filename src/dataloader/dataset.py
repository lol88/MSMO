import os
import numpy as np
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import cv2

index2label = {
               "us_fh_0": 76,
         "us_fh_1": 149,
    "camus_0": 1,
    "camus_1": 2,
    "camus_2": 3,
}

class SemiDataSets(Dataset):
    def __init__(
        self,
        args,
        base_dir=None,
        split="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self.args = args
        self.class_num = args.class_num
        self.dataset_name = args.dataset_name
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, self.dataset_name, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, self.dataset_name, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("{}:   total {} samples".format(self.split, len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, self.dataset_name, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, self.dataset_name, 'masks', case + '_mask.png'), cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']



        image = image.astype('float32')
        image = image.transpose(2, 0, 1)

        label = label.transpose(2, 0, 1)
        # label_onehot = np.zeros_like(label).astype('float32')
        # print(label.shape[-2])


        if self.class_num >= 2:
            label_onehot = np.zeros((self.class_num, label.shape[-2], label.shape[-1]), dtype="float32")
            for i in range(self.class_num):
                label_onehot[i][label[0] == index2label[f'{self.dataset_name}_{i}']] = 1

            # label[label == 76] = 1
            # label[label == 149] = 2
            # label_onehot = label
            # label_onehot = np.zeros((1, label.shape[-2], label.shape[-1]), dtype="float32")
            # for i in range(self.class_num):
            #     label_onehot[label[0] == index2label[f'{self.dataset_name}_{i}']] = i+1

                # cv2.imwrite(f"./test_{i}_mask.png", label_onehot[i] * 70)
                # print(np.unique(label_onehot[i]))
        else:
            label_onehot = label.astype('float32') / 255

        sample = {"image": image, "label": label_onehot, "idx": idx, "name": case}
        return sample


class SemiDataSets_sing(Dataset):
    def __init__(
        self,
        args,
        base_dir=None,
        split="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self.args = args
        self.class_num = args.class_num
        self.dataset_name = args.dataset_name
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, self.dataset_name, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, self.dataset_name, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, self.dataset_name, 'images', case + '.png'))

        label = \
        cv2.imread(os.path.join(self._base_dir, self.dataset_name, 'masks', case + '_mask.png'), cv2.IMREAD_GRAYSCALE)[
            ..., None]

        # print(np.unique(label))
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']


        image = image.astype('float32')
        image = image.transpose(2, 0, 1)

        # label = label.astype('float32') / 255

        label_one = np.zeros_like(label).astype('float32')
        label_one[label == int(self.args.ex_name)] = 1

        # onehot_mask = [label == i for i in [15, 38, 75, 113]]

        label_one = label_one.transpose(2, 0, 1)

        sample = {"image": image, "label": label_one, "idx": idx, "name": case}
        return sample



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

