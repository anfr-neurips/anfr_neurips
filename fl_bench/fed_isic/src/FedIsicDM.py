import os
import random

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from fl_bench.utils.misc import seed_everything


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FedIsicDataModule():
    def __init__(
        self,
        data_dir: str,
        client_idx: str,
        central: bool,
        batch_size: int = 64,
        seed: int = 0
    ):
        self.seed = seed
        seed_everything(seed)
        self.data_dir = data_dir
        self.client_idx = client_idx
        self.central = central
        self.batch_size = batch_size
        self.transform_train = A.Compose([
            A.RandomScale(0.07),
            A.Rotate(50),
            A.RandomBrightnessContrast(0.15, 0.1),
            A.Flip(p=0.5),
            A.Affine(shear=0.1),
            A.RandomCrop(200, 200),
            A.CoarseDropout(random.randint(1, 8), 16, 16),
            A.Normalize(always_apply=True),
            A.Resize(224, 224)
        ])
        self.transform_valid = A.Compose([
            A.CenterCrop(200, 200),
            A.Normalize(always_apply=True),
            A.Resize(224, 224)
        ])
        self.setup()

    def setup(self):

        self.train_dataset = FedIsic2019(
            data_dir=self.data_dir,
            center=self.client_idx,
            pooled=self.central,
            transform=self.transform_train
        )
        self.val_dataset = FedIsic2019(
            data_dir=self.data_dir,
            center=self.client_idx,
            pooled=self.central,
            train=False,
            transform=self.transform_valid
        )

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.viz_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}


class Isic2019Raw(Dataset):

    def __init__(
            self,
            data_dir,
            transform=None,
            X_dtype=torch.float32,
            y_dtype=torch.int64
    ):
        assert os.path.exists(data_dir)
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.transform = transform
        df = pd.read_csv(os.path.join(data_dir, "train_test_split"))
        self.image_paths = [
            os.path.join(
                data_dir,
                "ISIC_2019_Training_Input_preprocessed",
                image_name + ".jpg"
            )
            for image_name in df.image.tolist()
        ]
        self.targets = df.target
        self.centers = df.center

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=self.X_dtype),
            "label": torch.tensor(target, dtype=self.y_dtype)
        }


class FedIsic2019(Isic2019Raw):

    def __init__(
        self,
        data_dir,
        transform,
        center: str,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64
    ):

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            transform=transform,
            data_dir=data_dir,
        )
        # HACK: move from site-1 to site-6 to 0-5
        self.center = int(center.split('-')[-1]) - 1
        self.pooled = pooled
        self.train_test: str = "train" if train else "test"
        self.key: str = self.train_test + "_" + str(self.center)
        df = pd.read_csv(os.path.join(data_dir, "train_test_split"))
        # query = f"fold2 == '{self.key}'" if not self.pooled \
        #     else f"fold == '{self.train_test}'"
        # df2 = df.query(query).reset_index(drop=True)

        if self.pooled:
            df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)
        if not self.pooled:
            assert self.center in range(6)
            df2 = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)

        images = df2.image.tolist()
        self.image_paths = [
            os.path.join(
                data_dir,
                "ISIC_2019_Training_Input_preprocessed",
                image_name + ".jpg"
            )
            for image_name in images
        ]
        self.targets = df2.target
        self.centers = df2.center
