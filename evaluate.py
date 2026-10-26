"""
DeepShield — Evaluation & Benchmark Harness
Computes rigorous metrics on labeled test datasets:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, FPR
- Confusion Matrix [[TN, FP], [FN, TP]]
- Per-sample inference latency
Saves results to checkpoints/evaluation_results.json.
"""

import os
import argparse
import time
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from models import (
    DenseNet121Binary, ResNet50Binary,
    CHECKPOINT_DIR, DENSENET_CKPT_PATH, RESNET_CKPT_PATH,
    CLASS_AUTHENTIC, CLASS_AI, INFERENCE_TRANSFORM
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DeepShield.Evaluate")

RESULTS_PATH = os.path.join(CHECKPOINT_DIR, "evaluation_results.json")


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Computes exact Area Under the ROC Curve (ROC-AUC) via Mann-Whitney U statistic in NumPy.
    Requires no external dependencies (like scikit-learn).
    """
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank sorting
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_scores)) + 1.0

    # Handle ties with average rank
    unique_scores, counts = np.unique(y_scores, return_counts=True)
    for score, count in zip(unique_scores, counts):
        if count > 1:
            tie_mask = (y_scores == score)
            ranks[tie_mask] = np.mean(ranks[tie_mask])

    u_stat = np.sum(ranks[pos]) - (n_pos * (n_pos + 1.0)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """
    Computes classification evaluation metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, FPR, Confusion Matrix.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn

    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2.0 * (prec * rec) / max(1e-8, (prec + rec))
    fpr = fp / max(1, (fp + tn))
    auc = compute_roc_auc(y_true, y_scores)

    return {
        "accuracy": round(acc * 100.0, 2),
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "false_positive_rate": round(fpr * 100.0, 2),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]]
        },
        "sample_count": total
    }


@torch.no_grad()
def run_evaluation(data_dir: str = "data/test") -> Dict[str, Any]:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running DeepShield evaluation on device: {device}")

    if not os.path.isdir(data_dir):
        logger.warning(f"Evaluation directory '{data_dir}' does not exist.")
        # Return default evaluation baseline status if no test set is present
        baseline = {
            "status": "pending_test_dataset",
            "message": f"Test dataset not found at '{data_dir}'. Provide data/test/authentic and data/test/ai_generated to evaluate.",
            "metrics": None
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(baseline, f, indent=2)
        return baseline

    test_ds = datasets.ImageFolder(data_dir, transform=INFERENCE_TRANSFORM)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Initialize models
    densenet = DenseNet121Binary(pretrained_backbone=False).to(device).eval()
    if os.path.exists(DENSENET_CKPT_PATH):
        ckpt = torch.load(DENSENET_CKPT_PATH, map_location=device)
        densenet.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    resnet = ResNet50Binary(pretrained_backbone=False).to(device).eval()
    if os.path.exists(RESNET_CKPT_PATH):
        ckpt = torch.load(RESNET_CKPT_PATH, map_location=device)
        resnet.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    y_true = []
    dense_scores = []
    res_scores = []
    ensemble_scores = []
    latencies = []

    for images, labels in loader:
        images = images.to(device)
        label = labels.item()
        y_true.append(label)

        t0 = time.perf_counter()
        # DenseNet inference
        out_d = densenet(images)
        p_dense_ai = F.softmax(out_d, dim=1)[0, CLASS_AI].item()

        # ResNet inference
        out_r = resnet(images)
        p_res_ai = F.softmax(out_r, dim=1)[0, CLASS_AI].item()

        # Ensemble
        p_ensemble_ai = 0.50 * p_dense_ai + 0.50 * p_res_ai
        dt_ms = (time.perf_counter() - t0) * 1000.0

        dense_scores.append(p_dense_ai)
        res_scores.append(p_res_ai)
        ensemble_scores.append(p_ensemble_ai)
        latencies.append(dt_ms)

    y_true_np = np.array(y_true, dtype=int)
    ens_scores_np = np.array(ensemble_scores, dtype=float)
    ens_preds_np = (ens_scores_np >= 0.50).astype(int)

    metrics = compute_metrics(y_true_np, ens_preds_np, ens_scores_np)
    metrics["average_latency_ms"] = round(float(np.mean(latencies)), 2)

    results = {
        "status": "evaluated",
        "dataset_path": data_dir,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {RESULTS_PATH}")
    logger.info(f"Accuracy: {metrics['accuracy']}%, F1: {metrics['f1_score']}%, ROC-AUC: {metrics['roc_auc']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Evaluation Harness")
    parser.add_argument("--data_dir", type=str, default="data/test")
    args = parser.parse_args()
    run_evaluation(data_dir=args.data_dir)
