import os
import time
import json                             # ← 추가
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1) 데이터 증강 & 전처리 파이프라인
# ─────────────────────────────────────────────────────────────
# train_transform = transforms.Compose([
#     transforms.RandomRotation(20),
#     transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),          
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std =[0.229, 0.224, 0.225]
#     )
# ])
train_transform = transforms.Compose([
    # RandomResizedCrop을 512로
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
    transforms.Resize(540),         # 짧은 변을 540으로
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])
seed = 42

# ─────────────────────────────────────────────────────────────
# 2) ImageFolder 로더 + 클래스→인덱스 매핑
# ─────────────────────────────────────────────────────────────
data_root = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images'
full_dataset = datasets.ImageFolder(root=data_root, transform=None)
# train_ds = ImageFolder(root=data_root, transform=train_transform)
# val_ds   = ImageFolder(root=data_root, transform=val_transform)
style2idx   = full_dataset.class_to_idx.copy()
num_classes = len(style2idx)
print(f"Classes ({num_classes}): {style2idx}")

# ─────────────────────────────────────────────────────────────
# 3) train / val / test 분할
# ─────────────────────────────────────────────────────────────
n = len(full_dataset)
# n_train = int(0.8 * n)
# n_val   = int(0.1 * n)
# n_test  = n - n_train - n_val

# train_ds, val_ds, test_ds = random_split(
#     full_dataset,
#     [n_train, n_val, n_test],
#     generator=torch.Generator().manual_seed(42)
# )

# 3) 인덱스 섞기 + 분할
indices = np.arange(n)
np.random.seed(seed)
np.random.shuffle(indices)

n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train+n_val]
test_idx  = indices[n_train+n_val:]

# 4) Subset으로 train/val/test 생성
train_ds = Subset(datasets.ImageFolder(root=data_root, transform=train_transform),
                  train_idx)
val_ds   = Subset(datasets.ImageFolder(root=data_root, transform=val_transform),
                  val_idx)
test_ds  = Subset(datasets.ImageFolder(root=data_root, transform=val_transform),
                  test_idx)

print(f"Split sizes → train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

# ─────────────────────────────────────────────────────────────
# 4) DataLoader 생성
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 5) 모델 정의 (ResNet50 + custom head)
# ─────────────────────────────────────────────────────────────
class ResNet50Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        in_feats  = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.drop = nn.Dropout(0.2)
        self.fc1  = nn.Linear(in_feats, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(512, 128)
        self.fc3  = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)        # → (batch, 2048)
        x = self.drop(x)
        x = self.relu(self.fc1(x))  # → (batch, 1024)
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)             # → (batch, num_classes)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = ResNet50Head(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────
# 7) 학습 & 검증 루프 + JSON 로깅
# ─────────────────────────────────────────────────────────────
metrics = []                          # ← epoch별 메트릭을 담을 리스트

best_val_loss = float('inf')
num_epochs    = 50
print(f"Starting training for {num_epochs} epochs on device={device}")

for epoch in range(1, num_epochs+1):
    start_time = time.time()
    # — Training —
    model.train()
    train_loss = 0.0
    total      = 0
    for imgs_b, lbls_b in tqdm(train_loader, desc=f"[Epoch {epoch}] train", ncols=80):
        imgs_b, lbls_b = imgs_b.to(device, non_blocking=True), lbls_b.to(device, non_blocking=True)
        optimizer.zero_grad()
        out   = model(imgs_b)
        loss  = criterion(out, lbls_b)
        loss.backward()
        optimizer.step()

        bs = lbls_b.size(0)
        train_loss += loss.item() * bs
        total      += bs
    train_loss /= total

    # — Validation & Test —
    model.eval()
    val_loss = 0.0
    correct  = 0
    total_v  = 0
    correct_t= 0
    total_t  = 0
    with torch.no_grad():
        for imgs_b, lbls_b in val_loader:
            imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
            out = model(imgs_b)
            l = criterion(out, lbls_b)
            val_loss  += l.item() * lbls_b.size(0)
            correct   += (out.argmax(1) == lbls_b).sum().item()
            total_v   += lbls_b.size(0)
        for imgs_c, lbls_c in test_loader:
            imgs_c, lbls_c = imgs_c.to(device), lbls_c.to(device)
            out = model(imgs_c)
            correct_t += (out.argmax(1) == lbls_c).sum().item()
            total_t   += lbls_c.size(0)
    val_loss /= total_v
    val_acc   = correct / total_v
    test_acc  = correct_t / total_t

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}/{num_epochs}] "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
          f"test_acc={test_acc:.4f}  ({elapsed:.1f}s)")

    # → 메트릭 기록
    metrics.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss':   val_loss,
        'val_acc':    val_acc,
        'test_acc':   test_acc,
        'time_sec':   round(elapsed, 2)
    })

    # 최적 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_from_folder_pre.pth')
        print(f"→ New best model saved (val_loss={best_val_loss:.4f})\n")

# ─────────────────────────────────────────────────────────────
# 8) JSON 파일로 메트릭 저장
# ─────────────────────────────────────────────────────────────
with open('training_metrics.json', 'w') as fp:
    json.dump(metrics, fp, indent=2)

print("✅ Metrics saved to training_metrics_pre.json")
