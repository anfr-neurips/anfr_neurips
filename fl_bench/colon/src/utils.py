import json
import os
import pprint

import numpy as np
from dotenv import load_dotenv
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext

from fl_bench.utils.misc import seed_everything

load_dotenv()


def _get_site_class_summary(label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


class ColonSplitter(FLComponent):
    def __init__(
        self,
        split_dir,
        num_sites: int = 5,
        alpha: float = 0.5,
        seed: int = 0
    ):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        seed_everything(seed)

        if alpha < 0.0:
            raise ValueError(f"Alpha should be non-negative but was {alpha}.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(
            fl_ctx,
            f"Partition Colon training sets \
            into {self.num_sites} sites with Dirichlet \
            sampling under alpha {self.alpha}",
        )
        site_idx_train, class_sum_train = self._partition_data()

        # write to files
        if self.split_dir is None:
            raise ValueError(
                "You need to define a valid `split_dir` when splitting the data."
            )
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites}")
            sum_file.write(f"Dirichlet sampling parameter: {self.alpha}")
            sum_file.write("Train Class counts for each client:")
            sum_file.write(json.dumps(class_sum_train))
            pretty_json_str = pprint.pformat(
                json.dumps(class_sum_train), compact=True, indent=2
            ).replace("'", '"')
            sum_file.write(pretty_json_str)

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            site_file_name = site_file_path + str(site + 1) + "-train" + ".npy"
            np.save(site_file_name, np.array(site_idx_train[site]))

    def _partition_data(self):
        data_dir = os.getenv("COLON_ROOT")
        train_labels = np.load(f"{data_dir}/nct_wsi_100k_no_norm.npz", mmap_mode='r')["train_labels"].copy()
        # train_labels = colon_dataset["train_labels"]
        # convert to class encoding
        train_labels = np.argmax(train_labels, axis=1)

        min_size = 0
        K = 9
        N = train_labels.shape[0]
        site_idx_train = {}

        # split train
        while min_size < 10:
            idx_batch_train = [[] for _ in range(self.num_sites)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(self.alpha, self.num_sites))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / self.num_sites)
                        for p, idx_j in zip(proportions, idx_batch_train)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch_train = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch_train, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch_train])

        # shuffle
        for j in range(self.num_sites):
            np.random.shuffle(idx_batch_train[j])
            site_idx_train[j] = idx_batch_train[j]

        # collect class summaries
        class_sum_train = _get_site_class_summary(train_labels, site_idx_train)

        return site_idx_train, class_sum_train
