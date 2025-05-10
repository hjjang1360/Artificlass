import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────
# 1) 데이터 증강 & 전처리 파이프라인
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 2) ImageFolder 로더 + train/val/test 분할
# # ─────────────────────────────────────────────────────────────
# data_root = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_4'
# # data_root ='/home/hjjang/Artificlass/data_process/data/augmented_images_4'
train_root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/train'
val_root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/val'
test_root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/test'
# full_ds = datasets.ImageFolder(root=train_root, transform=None)
# val_full=datasets.ImageFolder(root=val_root, transform=None)
# style2idx = full_ds.class_to_idx.copy()
# num_classes = len(style2idx)
num_classes=7

# # 랜덤 인덱스 섞기
# n = len(full_ds)
# indices = np.arange(n)
# np.random.seed(seed)
# np.random.shuffle(indices)
# # n_train = int(0.8 * n)
# n_train=int(n)


# # n_val   = int(0.1 * n)
# # n_val=int(len(datasets.ImageFolder(root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/val', transform=None)))
# n_val=len(val_full)
# indices_v=np.arange(n_val)
# np.random.seed(seed)
# np.random.shuffle(indices_v)

# train_idx = indices[:n_train]
# val_idx   = indices_v[:n_val]
# test_idx  = indices_v[:n_val]

# # Subset & DataLoader
# train_ds = Subset(datasets.ImageFolder(root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/train', transform=train_transform), train_idx)
# val_ds   = Subset(datasets.ImageFolder(root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/val', transform=val_transform), val_idx)
# test_ds  = Subset(datasets.ImageFolder(root='/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/test', transform=val_transform), val_idx)

# loader_kwargs = dict(
#     batch_size=16,
#     num_workers=4,
#     pin_memory=True,
#     prefetch_factor=2,
#     persistent_workers=True
# )
# train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
# val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
# test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

train_ds = datasets.ImageFolder(root=train_root, transform=train_transform)
val_ds   = datasets.ImageFolder(root=val_root,   transform=val_transform)
test_ds  = datasets.ImageFolder(root=test_root,  transform=val_transform)

loader_kwargs = dict(
    batch_size=32,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

# ─────────────────────────────────────────────────────────────
# 4) 옵티마이저 & 손실함수 & (Optional) 스케줄러
# ─────────────────────────────────────────────────────────────
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────
# 3) 모델 정의 (EfficientNet B0 + custom head)
# ─────────────────────────────────────────────────────────────
class EfficientNetB0Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ImageNet pretrained EfficientNet-B0 불러오기
        backbone = models.efficientnet_b2(pretrained=True)
        in_feats = backbone.classifier[1].in_features
        # 기존 classifier 제거
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # 새 분류기 추가
        self.drop = nn.Dropout(0.2)
        self.fc1  = nn.Linear(in_feats, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(512, 128)
        self.fc3  = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)          # → (batch, in_feats)
        x = self.drop(x)
        x = self.relu(self.fc1(x))    # → (batch, 512)
        x = self.drop(x)
        x = self.relu(self.fc2(x))    # → (batch, 128)
        x = self.drop(x)
        return self.fc3(x)            # → (batch, num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = EfficientNetB0Head(num_classes).to(device)
# model = torch.compile(model)

# ─────────────────────────────────────────────────────────────
# 4) 옵티마이저 & 손실함수
# ─────────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# ─────────────────────────────────────────────────────────────
# 5) 학습 + Early Stopping 로직
# ─────────────────────────────────────────────────────────────
best_val_loss     = float('inf')
no_improve_epochs = 0
early_stop_patience = 5   # 5 에폭 연속 개선 없으면 멈춤
metrics = []

num_epochs = 50
print(f"Starting training for {num_epochs} epochs on device={device}")

for epoch in range(1, num_epochs+1):
    start_time = time.time()
    # — Training —
    model.train()
    train_loss = 0.0
    total      = 0
    for imgs_b, lbls_b in tqdm(train_loader, desc=f"[Epoch {epoch}] train", ncols=80):
        imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
        optimizer.zero_grad()
        with autocast():

            out  = model(imgs_b)
            loss = criterion(out, lbls_b)
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = lbls_b.size(0)
        train_loss += loss.item() * bs
        total      += bs
    train_loss /= total

    # — Validation & Test —
    model.eval()
    val_loss = 0.0; correct = 0; total_v = 0
    correct_t= 0; total_t = 0
    with torch.no_grad():
        for imgs_b, lbls_b in val_loader:
            imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
            out = model(imgs_b)
            l   = criterion(out, lbls_b)
            val_loss += l.item() * lbls_b.size(0)
            correct  += (out.argmax(1)==lbls_b).sum().item()
            total_v  += lbls_b.size(0)
        for imgs_c, lbls_c in test_loader:
            imgs_c, lbls_c = imgs_c.to(device), lbls_c.to(device)
            out = model(imgs_c)
            correct_t += (out.argmax(1)==lbls_c).sum().item()
            total_t   += lbls_c.size(0)
    val_loss /= total_v
    val_acc   = correct / total_v
    test_acc  = correct_t / total_t

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}/{num_epochs}] "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
          f"test_acc={test_acc:.4f}  ({elapsed:.1f}s)")

    # 메트릭 기록
    metrics.append({
        'epoch':     epoch,
        'train_loss':train_loss,
        'val_loss':  val_loss,
        'val_acc':   val_acc,
        'test_acc':  test_acc,
        'time_sec':  round(elapsed,2)
    })
    scheduler.step(val_loss)

    # Early Stopping 체크
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_model_efficientnet_b0_fixed.pth')
        print(f"→ New best model saved (val_loss={best_val_loss:.4f})\n")
    else:
        no_improve_epochs += 1
        print(f"  (no improvement for {no_improve_epochs}/{early_stop_patience} epochs)")

    if no_improve_epochs >= early_stop_patience:
        print(f"Early stopping triggered after epoch {epoch}")
        break

# ─────────────────────────────────────────────────────────────
# 6) JSON로 메트릭 저장
# ─────────────────────────────────────────────────────────────
with open('training_metrics_efficientnet_b0.json', 'w') as fp:
    json.dump(metrics, fp, indent=2)

print("✅ Done. Metrics saved to training_metrics_efficientnet_b0_fixed.json")
