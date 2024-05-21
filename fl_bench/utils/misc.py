import json
import os
import random
from typing import Dict

import numpy as np
import torch
from dotenv import load_dotenv
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

load_dotenv()


def load_weights(model: torch.nn.Module, weights: Dict[str, np.ndarray]) -> torch.nn.Module:
    local_var_dict = model.state_dict()
    model_keys = weights.keys()
    for var_name in local_var_dict:
        if var_name in model_keys:
            w = weights[var_name]
            try:
                local_var_dict[var_name] = torch.as_tensor(np.reshape(w, local_var_dict[var_name].shape))
            except Exception as e:
                raise ValueError(f"Convert weight from {var_name} failed with error {str(e)}")
    model.load_state_dict(local_var_dict)
    return model


def extract_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    local_state_dict = model.state_dict()
    local_model_dict = {}
    for var_name in local_state_dict:
        try:
            local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
        except Exception as e:
            raise ValueError(f"Convert weight from {var_name} failed with error: {str(e)}")
    return local_model_dict


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision('high')
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"

    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


class WeightedFocalLoss(_Loss):

    def __init__(
        self,
        alpha=torch.tensor(
            [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
        ),
        gamma=2.0,
    ):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()
