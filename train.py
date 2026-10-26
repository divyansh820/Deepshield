"""
DeepShield — Training & Fine-Tuning Pipeline
Supports training DenseNet-121 and ResNet-50 binary forensic classifiers.
Saves checkpoints to checkpoints/ directory with metadata.
"""

import os
import argparse
import time
import json
import logging
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image

from models import (
    DenseNet121Binary, ResNet50Binary,
    CHECKPOINT_DIR, DENSENET_CKPT_PATH, RESNET_CKPT_PATH,
    CLASS_AUTHENTIC, CLASS_AI, INFERENCE_TRANSFORM
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DeepShield.Train")


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    acc = (correct / max(1, total)) * 100.0
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    acc = (correct / max(1, total)) * 100.0
    return avg_loss, acc


def run_training(
    model_type: str = "ensemble",
    data_dir: str = "data",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    # Check for train / val directory structure
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        logger.warning(f"Data directories {train_dir} and/or {val_dir} not found.")
        logger.info(
            "Expected dataset format:\n"
            "  data/\n"
            "    train/\n"
            "      authentic/\n"
            "      ai_generated/\n"
            "    val/\n"
            "      authentic/\n"
            "      ai_generated/\n"
            "Supports datasets like GenImage, CIFAKE, and custom labeled collections."
        )
        return

    train_ds = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(val_dir, transform=INFERENCE_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    models_to_train = []
    if model_type in ["densenet121", "ensemble"]:
        models_to_train.append(("DenseNet-121", DenseNet121Binary(pretrained_backbone=True), DENSENET_CKPT_PATH))
    if model_type in ["resnet50", "ensemble"]:
        models_to_train.append(("ResNet-50", ResNet50Binary(pretrained_backbone=True), RESNET_CKPT_PATH))

    criterion = nn.CrossEntropyLoss()

    for name, model, ckpt_path in models_to_train:
        logger.info(f"=== Training {name} ===")
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0

        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
            scheduler.step()
            dt = time.time() - t0

            logger.info(
                f"[{name}] Epoch {ep:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | Time: {dt:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "version": "2.1.0",
                    "model_architecture": name,
                    "state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": ep
                }, ckpt_path)
                logger.info(f"Saved best {name} checkpoint to {ckpt_path} (Acc: {best_val_acc:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Model Fine-Tuning Pipeline")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "densenet121", "resnet50"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_training(
        model_type=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
