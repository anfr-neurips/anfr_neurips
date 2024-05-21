import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.nn import functional as F
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from fl_bench.utils.misc import seed_everything
import os
import json
import pandas as pd
import numpy as np

load_dotenv()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.):
        super().__init__()
        assert gamma >= 0
        self.register_buffer('weight', torch.tensor(weight))
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(
            input, target, reduction='none', weight=self.weight
        )
        p = torch.exp(-bce)
        loss = (1 - p) ** self.gamma * bce
        return loss.mean()


# splitter class, for use if we partition chest data a la CIFAR10
class ChestSplitter(FLComponent):
    def __init__(
        self,
        dataset: str,
        split_dir: str = "datalists",
        num_sites: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = os.getenv(f"{dataset}_ROOT")
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.seed = seed
        self.dest_path = os.path.join(
            self.data_dir, self.split_dir, f"seed_{self.seed}"
        )

        seed_everything(seed)

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            # if the split directory with this seed exists, skip the split
            if os.path.isdir(self.dest_path):
                self.log_info(
                    fl_ctx,
                    f"Directory {self.dest_path} exists, skipping the split.",
                )
                return

            self.split(fl_ctx)

    def split(self, fl_ctx):

        if fl_ctx:
            self.log_info(
                fl_ctx,
                f"Partitioning {self.dataset} training set \
                    into {self.num_sites} with seed = {self.seed}",
            )

        os.makedirs(self.dest_path, exist_ok=True)
        class_sum = self._partition_data()

        sum_file_name = os.path.join(self.dest_path, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Seed: {self.seed} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

    def _partition_data(self):

        if self.dataset == "CXR14":
            # load the data csv
            data = pd.read_csv(
                os.path.join(self.data_dir, "Data_Entry_2017_v2020.csv"),
                usecols=["Image Index", "Patient ID", "Finding Labels"],
            )
            # encode the labels and filter out all other columns
            labels = data["Finding Labels"].str.get_dummies(sep="|")
            data = pd.concat([data, labels], axis=1)
            data.drop(columns=["Finding Labels"], inplace=True)

            # filter the training/test rows
            train_list = pd.read_csv(
                os.path.join(self.data_dir, "train_val_list.txt"), header=None
            )[0].to_list()
            train_data = data[data["Image Index"].isin(train_list)]
            valid_list = pd.read_csv(
                os.path.join(self.data_dir, "test_list.txt"), header=None
            )[0].to_list()
            valid_data = data[data["Image Index"].isin(valid_list)]

            # get the unique patient IDs for the training set
            patients = train_data["Patient ID"].unique()
            # shuffle the patients
            np.random.shuffle(patients)
            # split the patients into self.num_sites
            site_idx = np.array_split(patients, self.num_sites)

            # prepare the decathlon datalist json files for each site
            # with their own training set and the commmon validation set
            for i, site in enumerate(site_idx):
                site_list = train_data[train_data["Patient ID"].isin(site)]
                json_data = {"training": [], "validation": []}
                valid_list = valid_data
                for idx, row in site_list.iterrows():
                    new_item = {}
                    new_item["image"] = row["Image Index"]
                    new_item["label"] = row.iloc[2:].values.tolist()
                    json_data["training"].append(new_item)
                for idx, row in valid_list.iterrows():
                    new_item = {}
                    new_item["image"] = row["Image Index"]
                    new_item["label"] = row.iloc[2:].values.tolist()
                    json_data["validation"].append(new_item)
                with open(f"{self.dest_path}/site-{i+1}.json", "w") as f:
                    json.dump(json_data, f, indent=4)

            # get the class counts for each site
            class_sum = {}
            for i, site in enumerate(site_idx):
                class_sum[f"site-{i+1}"] = (
                    data[data["Patient ID"].isin(site)].iloc[:, 2:].sum().to_dict()
                )
                # add the number of samples to the summary, which is the number
                # of patients multiplied by the number of images per patient
                cnt = 0
                for patient in site:
                    cnt += len(data[data["Patient ID"] == patient])
                class_sum[f"site-{i+1}"]["samples"] = cnt

            return class_sum
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
