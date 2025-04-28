import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

# --- Settings ---
data_dir = './augmented_images'
batch_size = 32
val_split = 0.2
num_epochs = 30
num_classes = 7
lr = 1e-4
weight_decay = 1e-4  # L2 reg equivalent
patience_es = 5
patience_lr = 3
min_lr = 1e-7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transforms ---
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- Dataset & DataLoader ---
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
num_total = len(full_dataset)
num_val = int(val_split * num_total)
num_train = num_total - num_val
train_dataset, val_dataset = random_split(
    full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42)
)
# Override val dataset transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True, prefetch_factor=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True, prefetch_factor=2
)

# --- Model Definition ---
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rates=(0.5, 0.3)):
        super().__init__()
        # Backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits  # use CrossEntropyLoss internally

model = ResNet50Classifier(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=patience_lr, min_lr=min_lr, verbose=True
)

# --- Training Loop ---
history = {
    'epoch': [], 'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}
best_val_loss = float('inf')
epochs_no_improve = 0
start_time = time.time()

for epoch in range(1, num_epochs+1):
    # Train
    model.train()
    running_loss = 0.0
    correct = total = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Train"):  
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # Validate
    model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Val  "):  
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / total
    val_acc = correct / total

    # Scheduler step
    scheduler.step(val_loss)

    # EarlyStopping & Checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience_es:
            print(f"Early stopping at epoch {epoch}")
            break

    # Record history
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Ep{epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, " +
          f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

# --- Save History ---
history_df = pd.DataFrame(history)
history_df.to_csv('training_history.csv', index=False)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
plt.plot(history['epoch'], history['val_acc'],   label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig('accuracy_plot.png'); plt.close()

plt.figure(figsize=(8,6))
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'],   label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig('loss_plot.png'); plt.close()

print("✅ 학습 완료. best_model.pth 와 training_history.csv 파일이 저장되었습니다.")
