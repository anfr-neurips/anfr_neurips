import json
import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ChestDataModuleTest:
    def __init__(
        self,
        data_dir: str,
        client_idx: str,
        cache_rate: float = 0.0,
    ):

        self.cache_rate = cache_rate

        self.transform_valid = A.Compose([
            A.Normalize(mean=0.5, std=0.25, always_apply=True),
        ])

        datalist_paths = os.path.join(data_dir, "datalists")
        client_datalist = os.path.join(
            datalist_paths, client_idx + "_test.json"
        )

        self.test_list = json.load(open(client_datalist, "r"))["testing"]

        self.setup()

    def setup(self):

        self.test_dataset = CacheDataset(
            data=self.test_list,
            transform=self.transform_valid,
            cache_rate=self.cache_rate,
            num_workers=4,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=100, shuffle=False, num_workers=4
        )

    def teardown(self):
        self.test_dataset = {}
        self.test_dataloader = {}


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
