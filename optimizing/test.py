import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# -----------------------------
# 0) 설정
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# 1) 데이터 로더 준비
# -----------------------------
# train_val_dir = './augmented_images_512'
# test_dir      = './augmented_images_test_512'

train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(512, scale=(0.8,1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])
val_transform = transforms.Compose([
    transforms.Resize(540),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])
seed = 42

# 학습+검증 데이터셋 분할
data_root = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_4'
# data_root ='/home/hjjang/Artificlass/data_process/data/augmented_images_4'
full_ds = datasets.ImageFolder(root=data_root, transform=None)
style2idx = full_ds.class_to_idx.copy()
num_classes = len(style2idx)

# 랜덤 인덱스 섞기
n = len(full_ds)
indices = np.arange(n)
np.random.seed(seed)
np.random.shuffle(indices)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train+n_val]
test_idx  = indices[n_train+n_val:]

# Subset & DataLoader
train_ds = Subset(datasets.ImageFolder(root=data_root, transform=train_transform),
                  train_idx)
val_ds   = Subset(datasets.ImageFolder(root=data_root, transform=val_transform),
                  val_idx)
test_ds  = Subset(datasets.ImageFolder(root=data_root, transform=val_transform),
                  test_idx)

loader_kwargs = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

# num_classes = len(full_ds.classes)

# -----------------------------
# 2) 모델 정의
# -----------------------------
class ConvResNet512(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 초기 Conv 레이어들
        self.conv1 = nn.Conv2d(3,3,3,stride=2,padding=1)
        self.bn1   = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3,3,3,stride=2,padding=1)
        self.bn2   = nn.BatchNorm2d(3)
        self.resize= nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(3,3,3,padding=1)
        self.bn3   = nn.BatchNorm2d(3)
        self.relu  = nn.ReLU(inplace=True)

        # ResNet50 백본
        backbone = models.resnet50(pretrained=True)
        in_feats = backbone.fc.in_features   # ← save this first!
        backbone.fc = nn.Identity()           # then remove the old classifier
        self.backbone = backbone

        # 커스텀 헤드
        self.drop1 = nn.Dropout(0.5)
        self.fc1   = nn.Linear(in_feats, 512)
        self.bn_fc1= nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(512, 128)
        self.bn_fc2= nn.BatchNorm1d(128)
        self.fc3   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.resize(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.backbone(x)  # (batch,2048)
        x = self.drop1(x)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop2(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

model = ConvResNet512(num_classes).to(device)

# -----------------------------
# 3) 손실 함수, 옵티마이저, 스케줄러
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# -----------------------------
# 4) 학습 루프 + EarlyStopping
# -----------------------------
num_epochs = 30
best_val_loss = float('inf')
no_improve = 0
patience = 5

history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

for epoch in range(1, num_epochs+1):
    # Training
    model.train()
    running_loss = 0; correct=0; total=0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * lbls.size(0)
        preds = out.argmax(1)
        correct += (preds==lbls).sum().item()
        total += lbls.size(0)
    train_loss = running_loss/total
    train_acc  = correct/total

    # Validation
    model.eval()
    running_loss = 0; correct=0; total=0
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            running_loss += loss.item()*lbls.size(0)
            preds = out.argmax(1)
            correct += (preds==lbls).sum().item()
            total += lbls.size(0)
    val_loss = running_loss/total
    val_acc  = correct/total

    # 기록
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")

    # Scheduler step
    scheduler.step(val_loss)

    # 체크포인트 & EarlyStopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model_512_conv.pth')
        print("  ** Saved Best Model **")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# # -----------------------------
# # 5) 학습 결과 저장 & 테스트 평가
# # -----------------------------
# # 학습 기록 CSV
# pd.DataFrame(history).to_csv('training_history_512_conv.csv', index=False)

# # 테스트 평가
# model.load_state_dict(torch.load('best_model_512_conv.pth'))
# model.eval()
# running_loss=0; correct1=0; correct5=0; total=0
# with torch.no_grad():
#     for imgs, lbls in tqdm(test_loader, desc="Test Eval"):
#         imgs, lbls = imgs.to(device), lbls.to(device)
#         out = model(imgs)
#         loss=criterion(out, lbls)
#         running_loss += loss.item()*lbls.size(0)
#         probs = F.softmax(out, dim=1)
#         _, pred1 = probs.topk(1, dim=1)
#         _, pred5 = probs.topk(5, dim=1)
#         correct1 += (pred1.view(-1)==lbls).sum().item()
#         correct5 += sum([lbls[i] in pred5[i] for i in range(lbls.size(0))])
#         total += lbls.size(0)
# test_loss = running_loss/total
# test_acc  = correct1/total
# test_top5 = correct5/total
# print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}, top5 {test_top5:.4f}")

# # -----------------------------
# # 6) 학습 곡선 저장
# # -----------------------------
# plt.figure(figsize=(8,6))
# plt.plot(history['train_acc'], label='Train Acc')
# plt.plot(history['val_acc'],   label='Val Acc')
# plt.xlabel('Epoch'); plt.ylabel('Accuracy')
# plt.title('Accuracy over Epochs'); plt.legend(); plt.grid(True)
# plt.savefig('accuracy_plot_512_conv.png')
# plt.close()

# plt.figure(figsize=(8,6))
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'],   label='Val Loss')
# plt.xlabel('Epoch'); plt.ylabel('Loss')
# plt.title('Loss over Epochs'); plt.legend(); plt.grid(True)
# # plt.savefig('loss_plot_512_conv.png')
# plt.close()

print("✅ Training, evaluation, and plotting complete!")
