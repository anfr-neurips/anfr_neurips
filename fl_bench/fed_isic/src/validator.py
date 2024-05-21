from typing import Any, Dict

import torch
from monai.inferers.inferer import SimpleInferer
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator(object):
    def __init__(self):
        self.metric_score = 0.0
        self.inferer = SimpleInferer()
        self.valid_metric = balanced_accuracy_score

    def validate_loop(self, model, data_loader) -> Dict[str, Any]:
        # Run inference over whole validation set

        y = []
        y_pred = []

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    batch["image"] = batch["image"].to("cuda:0")
                    batch["label"] = batch["label"]
                    y.append(batch["label"])
                    batch["preds"] = self.inferer(batch["image"], model)
                    _, pred_label = torch.max(batch["preds"].data, 1)
                    y_pred.append(pred_label)

                y = torch.cat(y)
                y_pred = torch.cat(y_pred)
                self.metric_score = self.valid_metric(y.cpu(), y_pred.cpu())

        # Collect metrics
        # raw_metrics = self.metric.aggregate()
        # self.metric.reset()

        metrics = {"balanced_accuracy": self.metric_score}
        # for organ, idx in self.fg_classes.items():
        #     mean += raw_metrics[idx]
        #     metrics["val_meandice_" + organ] = raw_metrics[idx]
        # metrics["val_meandice"] = mean / len(self.fg_classes)

        # for k, v in metrics.items():
        #     if isinstance(v, torch.Tensor):
        #         metrics[k] = v.tolist()
        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        return self.validate_loop(model, data_loader)
