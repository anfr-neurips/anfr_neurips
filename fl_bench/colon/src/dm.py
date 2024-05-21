import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from fl_bench.utils.misc import seed_everything


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ColonDataModule:
    def __init__(
        self,
        data_dir: str,
        idx_root: str,
        client_idx: str,
        central: bool,
        batch_size: int = 64,
        seed: int = 0,
    ):
        self.seed = seed
        seed_everything(seed)
        self.data_dir = data_dir
        self.client_idx = client_idx
        self.idx_root = idx_root
        self.central = central
        self.batch_size = batch_size
        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2),
                transforms.RandomErasing(value='random')
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.target_transform = transforms.Compose([lambda x: x.squeeze(-1)])
        self.setup()

    def setup(self):

        if self.central is False:
            site_idx_train_name = os.path.join(
                self.idx_root, self.client_idx + "-train" + ".npy"
            )

            if os.path.exists(site_idx_train_name):
                site_idx_train = np.load(site_idx_train_name).tolist()
            else:
                raise ValueError(
                    f"File {site_idx_train_name} does not exist!"
                )
        else:
            site_idx_train = None

        npz_file = os.path.join(self.data_dir, "nct_wsi_100k_no_norm.npz")

        whole_tset = ColonDataset(
            npz_file=npz_file,
            part="train",
            indices=site_idx_train,
            transform=self.transform_train,
            target_transform=self.target_transform,
        )

        self.val_dataset = ColonDataset(
            npz_file=npz_file,
            part="val",
            transform=self.transform_valid,
            target_transform=self.target_transform,
        )

        self.train_dataset = whole_tset
        # self.train_dataset = (
        #     torch.utils.data.Subset(whole_tset, site_idx_train)
        #     if site_idx_train is not None
        #     else whole_tset
        # )

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}


class ColonDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file, indices=None,  part="train", transform=None, target_transform=None):
        if indices:
            self.images = np.load(npz_file, mmap_mode='r')[f"{part}_images"][indices].copy()
            self.labels = np.load(npz_file, mmap_mode='r')[f"{part}_labels"][indices].copy()
        else:
            self.images = np.load(npz_file, mmap_mode='r')[f"{part}_images"].copy()
            self.labels = np.load(npz_file, mmap_mode='r')[f"{part}_labels"].copy()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        # back to hwc so ToTensor can operate
        image = np.transpose(image, (1, 2, 0))
        label = self.labels[index]
        # from one-hot to categorical
        label = np.argmax(label, keepdims=True)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {"image": image, "label": label}
