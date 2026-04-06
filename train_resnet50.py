"""
ThreadAI - Training Script

Trains ResNet50 models for:
1. Clothing Category Classification
2. Clothing Style Classification

Features:
- Stratified train/val split
- Class imbalance handling (weighted loss)
- Transfer learning (freeze → unfreeze)
- Early stopping + best model saving
- Evaluation: classification report, confusion matrix, ROC-AUC

Dataset: Myntra scraped dataset (8k+ images)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from collections import Counter
from tqdm import tqdm


# CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_merged")

BATCH_SIZE = 32
EPOCHS = 25                 
FREEZE_EPOCHS = 3           
LR_HEAD = 1e-4
LR_FULL = 1e-5
PATIENCE = 5                
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# TRANSFORMS

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# DATASET (LABEL SOURCE OF TRUTH)

dataset = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = dataset.classes
num_classes = len(class_names)

print("\n class_to_idx:", dataset.class_to_idx)


# STRATIFIED SPLIT

indices_per_class = {i: [] for i in range(num_classes)}
for idx, (_, lbl) in enumerate(dataset.samples):
    indices_per_class[lbl].append(idx)

train_idx, val_idx = [], []
for lbl, indices in indices_per_class.items():
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx.extend(indices[:split])
    val_idx.extend(indices[split:])

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# CLASS WEIGHTS

train_labels = [dataset.samples[i][1] for i in train_idx]
counts = Counter(train_labels)
class_weights = torch.tensor(
    [1.0 / counts[i] for i in range(num_classes)],
    dtype=torch.float
).to(DEVICE)

print(" Class weights:", class_weights.cpu().numpy())


# MODEL

from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_HEAD)


# TRAINING WITH AUTO CONTROL

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch+1}/{EPOCHS}")


    if epoch == FREEZE_EPOCHS:
        print("] Unfreezing full model")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)


    model.train()
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ---- VALIDATION 
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    val_acc = report["accuracy"]

    print(f" Val Accuracy: {val_acc:.4f}")

    # ---- COLLAPSE DETECTION
    dominant_ratio = Counter(y_pred).most_common(1)[0][1] / len(y_pred)
    if dominant_ratio > 0.8:
        print(" Collapse detected — stopping training")
        break

    # ---- EARLY STOPPING 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0

        torch.save(
            {
                "model_state": model.state_dict(),
                "idx_to_class": idx_to_class
            },
            os.path.join(BASE_DIR, "resnet50_best.pth")
        )
        print(" Saved BEST model")
    else:
        epochs_no_improve += 1
        print(f" No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(" Early stopping triggered")
            break

print("\n Training complete")
print(f" Best validation accuracy: {best_val_acc:.4f}")

# ---- DETAILED EVALUATION
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import numpy as np

model.eval()

y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)

        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        preds = outputs.argmax(1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        y_probs.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
try:
    roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    print("ROC-AUC Score:", round(roc_auc, 4))
except:
    print("ROC-AUC could not be computed")
cm = confusion_matrix(y_true, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Normalized Confusion Matrix")
plt.show()
