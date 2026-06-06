"""
Intelligent RoadEye — VGG16 Training Pipeline (PyTorch)
Optimized for NVIDIA RTX 3050 4GB — CUDA enabled

Dataset: data/merged/crack/ and data/merged/pothole/
Run merge_dataset.py first.

Output: model/roadeye_vgg16.pth
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "data_dir"      : os.path.join(BASE_DIR, "data", "merged"),
    "model_dir"     : os.path.join(BASE_DIR, "model"),
    "model_path"    : os.path.join(BASE_DIR, "model", "roadeye_vgg16.pth"),
    "class_idx_path": os.path.join(BASE_DIR, "model", "class_indices.json"),
    "img_size"      : 128,
    "batch_size"    : 32,      # PyTorch handles 32 fine on RTX 3050
    "epochs"        : 20,
    "learning_rate" : 1e-4,
    "val_split"     : 0.2,
    "num_workers"   : 0,       # 0 for Windows — avoids multiprocessing issues
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ── Data Transforms ───────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
def build_dataloaders():
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"])

    # Save class indices
    with open(CONFIG["class_idx_path"], "w") as f:
        json.dump(full_dataset.class_to_idx, f, indent=2)
    print(f"Classes: {full_dataset.class_to_idx}")

    # Split train / val
    total     = len(full_dataset)
    val_size  = int(total * CONFIG["val_split"])
    train_size= total - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms separately
    train_ds.dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=train_transforms)
    val_ds_copy      = datasets.ImageFolder(CONFIG["data_dir"], transform=val_transforms)

    # Rebuild with correct indices
    train_indices = train_ds.indices
    val_indices   = val_ds.indices

    from torch.utils.data import Subset
    train_subset = Subset(
        datasets.ImageFolder(CONFIG["data_dir"], transform=train_transforms),
        train_indices
    )
    val_subset = Subset(
        datasets.ImageFolder(CONFIG["data_dir"], transform=val_transforms),
        val_indices
    )

    train_loader = DataLoader(
        train_subset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True if device.type == "cuda" else False,
    )

    return train_loader, val_loader, full_dataset.class_to_idx


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze early layers — only train last 3 conv blocks + classifier
    for i, layer in enumerate(model.features):
        if i < 17:
            for param in layer.parameters():
                param.requires_grad = False

    # AdaptiveAvgPool ensures classifier works with any input size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # Classifier — 512 features from AdaptiveAvgPool
    model.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

    return model.to(device)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"loss: {total_loss/(batch_idx+1):.4f} "
                  f"acc: {correct/total:.4f}", end="\r")

    return total_loss / len(loader), correct / total


# ── Validation Loop ───────────────────────────────────────────────────────────
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Plot History ──────────────────────────────────────────────────────────────
def plot_history(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_accs, label="Train")
    axes[0].plot(val_accs,   label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_losses, label="Train")
    axes[1].plot(val_losses,   label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("RoadEye VGG16 — Training History")
    plt.tight_layout()
    path = os.path.join(CONFIG["model_dir"], "training_history.png")
    plt.savefig(path)
    plt.close()
    print(f"Training history saved: {path}")


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(all_preds, all_labels, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — RoadEye VGG16")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(CONFIG["model_dir"], "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Intelligent RoadEye — VGG16 Training (PyTorch)")
    print("=" * 60)

    if not os.path.exists(CONFIG["data_dir"]):
        print(f"\n[ERROR] Data not found: {CONFIG['data_dir']}")
        print("Run model/merge_dataset.py first!")
        sys.exit(1)

    crack_count   = len(os.listdir(os.path.join(CONFIG["data_dir"], "crack")))
    pothole_count = len(os.listdir(os.path.join(CONFIG["data_dir"], "pothole")))

    print(f"\nDataset:")
    print(f"  Crack   : {crack_count}")
    print(f"  Pothole : {pothole_count}")
    print(f"  Total   : {crack_count + pothole_count}")
    print(f"\nConfig:")
    print(f"  Image size  : {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"  Batch size  : {CONFIG['batch_size']}")
    print(f"  Epochs      : {CONFIG['epochs']}")
    print(f"  Device      : {device}")

    # Build data
    train_loader, val_loader, class_to_idx = build_dataloaders()
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    num_classes  = len(class_to_idx)
    class_names  = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # Build model
    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training
    print("\n── Starting Training ──────────────────────────────")
    best_val_acc   = 0.0
    patience_count = 0
    patience_limit = 6

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, all_preds, all_labels = val_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - start_time

        print(f"\nEpoch {epoch:2d}/{CONFIG['epochs']} "
              f"| train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
              f"| val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
              f"| {epoch_time:.0f}s | elapsed: {elapsed/60:.1f}min")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"        : epoch,
                "model_state"  : model.state_dict(),
                "optimizer"    : optimizer.state_dict(),
                "val_acc"      : val_acc,
                "class_to_idx" : class_to_idx,
            }, CONFIG["model_path"])
            print(f"  ✓ Best model saved (val_acc: {val_acc:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation
    total_time = (time.time() - start_time) / 60
    print(f"\n── Training Complete ──────────────────────────────")
    print(f"  Best val accuracy : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Total time        : {total_time:.1f} minutes")
    print(f"  Model saved       : {CONFIG['model_path']}")

    # Plots
    plot_history(train_losses, val_losses, train_accs, val_accs)

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    plot_confusion_matrix(all_preds, all_labels, class_names)

    print("\nRun: python backend/app.py")
