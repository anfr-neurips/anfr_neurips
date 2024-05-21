import numpy as np
import torch
from monai.transforms import Activations
from sklearn.metrics import (multilabel_confusion_matrix, roc_auc_score,
                             roc_curve)
from tqdm import tqdm

cls_map = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "No Finding",
    "Pneumonia",
    "Pneumothorax"
]


class Validator:
    def __init__(self):
        self.transform_post = Activations(sigmoid=True)

    def run(self, model, data_loader, logger=None, step=None):

        model.eval()
        mean = 0.0
        metrics = {}
        y = []
        y_pred = []

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch in tqdm(
                    data_loader,
                    desc="Validation DataLoader", dynamic_ncols=True
                ):
                    batch["image"] = batch["image"].to("cuda:0")
                    batch["label"] = batch["label"].to("cuda:0")
                    batch["preds"] = model(batch["image"])
                    if self.transform_post is not None:
                        batch["preds"] = self.transform_post(batch["preds"])
                    y.append(batch["label"])
                    y_pred.append(batch["preds"])

                y = torch.cat(y)
                y_pred = torch.cat(y_pred)

                if logger is not None:
                    for idx in range(y_pred.shape[1]):
                        logger.add_pr_curve(
                            f"pr_curve_disease_{idx}",
                            labels=y[:, idx],
                            predictions=y_pred[:, idx],
                        )

        # if y or y_pred contain nans, skip all metrics and return a dict with 0 for all metrics
        if torch.isnan(y).any() or torch.isnan(y_pred).any():
            for cls in cls_map:
                metrics["auroc_" + cls] = 0
            metrics["mean_auroc"] = 0
            metrics["macro_f1_score"] = 0
            metrics["macro_accuracy"] = 0
            metrics["macro_specificity"] = 0
            metrics["macro_sensitivity"] = 0
            metrics["macro_precision"] = 0
            metrics["optimal_thresholds"] = [0] * y_pred.shape[1]
            return metrics        
        y_pred_np = y_pred.numpy(force=True)
        y_np = y.numpy(force=True)

        scores = roc_auc_score(y_np, y_pred_np, average=None)

        for i, score in enumerate(scores):
            mean += score
            metrics["auroc_" + cls_map[i]] = score
        metrics["mean_auroc"] = mean / len(scores)

        # Youden's J statistic
        # to find the optimal threshold for each class
        optimal_thresholds = np.zeros(y.shape[1])
        for class_idx in range(y_np.shape[1]):
            fpr, tpr, thresholds = roc_curve(
                y_np[:, class_idx], y_pred_np[:, class_idx]
            )
            optimal_thresholds[class_idx] = thresholds[np.argmax(tpr - fpr)]
        bin_labels = (y_pred_np > optimal_thresholds).astype(np.int32)

        # Metrics calculation (macro) over the whole set
        total_cm = multilabel_confusion_matrix(y_true=y_np, y_pred=bin_labels)
        eps = 1e-7
        f1 = []
        accuracy = []
        specificity = []
        sensitivity = []
        precision = []
        for cls_cm in total_cm:
            TP = cls_cm[1, 1]
            TN = cls_cm[0, 0]
            FP = cls_cm[0, 1]
            FN = cls_cm[1, 0]
            f1.append(2 * TP / (2 * TP + FN + FP + eps))
            accuracy.append((TP + TN) / (TP + TN + FP + FN + eps))
            specificity.append(TN / (TN + FP + eps))
            sensitivity.append(TP / (TP + FN + eps))
            precision.append(TP / (TP + FP + eps))

        metrics["macro_f1_score"] = np.mean(f1)
        metrics["macro_accuracy"] = np.mean(accuracy)
        metrics["macro_specificity"] = np.mean(specificity)
        metrics["macro_sensitivity"] = np.mean(sensitivity)
        metrics["macro_precision"] = np.mean(precision)
        metrics["optimal_thresholds"] = list(optimal_thresholds)

        return metrics
