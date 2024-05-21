import json
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ChestDataModule:
    def __init__(
        self,
        data_dir: str,
        client_idx: str,
        central: bool,
        batch_size: int = 64,
        cache_rate: float = 1.0,
        seed: int = 0,
    ):
        self.seed = seed
        # seed_everything(seed)
        self.batch_size = batch_size
        self.cache_rate = cache_rate

        self.transform_train = A.Compose([
            A.ShiftScaleRotate(
                shift_limit_x=0.1,
                shift_limit_y=0.1,
                border_mode=0,
                value=np.random.randint(0, 255),
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=3,
                max_height=70,
                max_width=70,
                min_holes=1,
                min_height=30,
                min_width=30,
                fill_value=np.random.randint(0, 255),
                p=0.5,
            ),
            A.RandomCrop(200, 200),
            A.Resize(224, 224),
            A.Normalize(mean=0.5, std=0.25, always_apply=True),
        ])

        self.transform_valid = A.Compose([
            A.Normalize(mean=0.5, std=0.25, always_apply=True),
        ])

        datalist_paths = os.path.join(data_dir, "datalists")
        if central:
            client_datalist = os.path.join(datalist_paths, "client_All.json")
        else:
            client_datalist = os.path.join(
                datalist_paths, client_idx + ".json"
            )

        # read the class counts from the summary file
        summary_file = os.path.join(datalist_paths, client_idx + "_summary.txt")
        with open(summary_file, "r") as f:
            summary = f.read()
        class_counts = json.loads(summary)
        class_counts = list(class_counts.values())
        self.weights = class_counts[:-1]
        num_samples = class_counts[-1]
        self.weights = num_samples / (np.array(self.weights) + 1e-6) - 1

        self.train_list = json.load(open(client_datalist, "r"))["training"]
        self.valid_list = json.load(open(client_datalist, "r"))["validation"]

        self.setup()

    def setup(self):

        self.train_dataset = CacheDataset(
            data=self.train_list,
            transform=self.transform_train,
            cache_rate=self.cache_rate,
            num_workers=6,
        )
        self.valid_dataset = CacheDataset(
            data=self.valid_list,
            transform=self.transform_valid,
            cache_rate=self.cache_rate,
            num_workers=6,
        )

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            # worker_init_fn=seed_worker,
            # generator=g,
        )
        self.val_dataloader = DataLoader(
            self.valid_dataset, batch_size=100, shuffle=False, num_workers=6
        )

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}


class CacheDataset(Dataset):

    def __init__(self, data, transform, cache_rate=1.0, num_workers=1):

        self.data = data
        self.transform = transform
        self.set_rate = cache_rate
        self.num_workers = num_workers
        self.cache_num = 0
        self._cache = []
        self.set_data(data)

    def set_data(self, data):

        self.data = data
        lgth = len(self.data)
        self.cache_num = min(int(lgth * self.set_rate), lgth)
        indices = list(range(self.cache_num))
        with ThreadPool(self.num_workers) as p:
            self._cache = list(
                tqdm(
                    p.imap(self._load_cache_item, indices),
                    total=len(indices),
                    desc="Loading dataset"
                )
            )

        return

    def _load_cache_item(self, idx: int):
        image_path = self.data[idx]['image']
        image = np.array(Image.open(image_path))
        label = self.data[idx]['label']
        item = {
            "image": np.ascontiguousarray(image),
            "label": np.ascontiguousarray(label)
        }

        return item

    def _transform(self, index: int):

        if index % len(self) < self.cache_num:
            # data has been cached and is a dict
            data = self._cache[index]
            # deepcopy so we don't mess with the cache
            data = deepcopy(data)
            if self.transform is not None:
                augmented = self.transform(image=data['image'])
                image = augmented["image"]
            image = np.expand_dims(image, 0).astype(np.float32)

            return {
                "image": torch.from_numpy(image),
                "label": torch.tensor(data['label'])
            }
        else:
            # data has not been cached
            image_path = self.data[index]['image']
            image = np.array(Image.open(image_path))
            label = self.data[index]['label']

            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]
            image = np.expand_dims(image, 0).astype(np.float32)

            return {
                "image": torch.from_numpy(image),
                "label": torch.tensor(label)
            }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self._transform(index)
