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

# ─────────────────────────────────────────────────────────────
# 1) 데이터 파이프라인 정의 (생략, 이전과 동일)
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
# 2) Dataset / DataLoader 정의 (생략, 이전과 동일)
# ─────────────────────────────────────────────────────────────
data_root = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_4'
full_ds = datasets.ImageFolder(root=data_root, transform=None)
style2idx = full_ds.class_to_idx.copy()
num_classes = len(style2idx)

# train/val/test split
n = len(full_ds)
indices = np.arange(n)
np.random.seed(seed)
np.random.shuffle(indices)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train+n_val]
test_idx  = indices[n_train+n_val:]

train_ds = Subset(datasets.ImageFolder(root=data_root, transform=train_transform), train_idx)
val_ds   = Subset(datasets.ImageFolder(root=data_root, transform=val_transform), val_idx)
test_ds  = Subset(datasets.ImageFolder(root=data_root, transform=val_transform), test_idx)
print(f"Split sizes → train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
loader_kwargs = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

# ─────────────────────────────────────────────────────────────
# 3) EfficientNet-B0 + head 정의
# ─────────────────────────────────────────────────────────────
class EfficientNetB0Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b0(pretrained=True)
        in_feats = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.drop = nn.Dropout(0.2)
        self.fc1  = nn.Linear(in_feats, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(512, 128)
        self.fc3  = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = EfficientNetB0Head(num_classes).to(device)

# ─────────────────────────────────────────────────────────────
# 4) 옵티마이저 & 손실함수 준비
#    - phase1: backbone+head
#    - phase2: head only
# ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
opt_all  = optim.Adam(model.parameters(), lr=1e-4)
head_params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters())
opt_head = optim.Adam(head_params, lr=1e-4)

# ─────────────────────────────────────────────────────────────
# 5) 두 단계 학습 루프 + Early Stopping
# ─────────────────────────────────────────────────────────────
metrics = []
best_val_loss      = float('inf')
no_improve_epochs  = 0
early_stop_patience = 5   # 개선 없으면 5 에폭 만에 중단

total_epochs       = 50
freeze_after_epoch = 10

for epoch in range(1, total_epochs+1):
    start_time = time.time()
    phase = 'all' if epoch <= freeze_after_epoch else 'head'
    optimizer = opt_all if phase=='all' else opt_head

    # phase2 진입 시 backbone freeze
    if epoch == freeze_after_epoch+1:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # — Training —
    model.train()
    train_loss = 0.0; total=0
    for imgs_b, lbls_b in tqdm(train_loader, desc=f"[Epoch {epoch}] train({phase})", ncols=80):
        imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
        optimizer.zero_grad()
        out  = model(imgs_b)
        loss = criterion(out, lbls_b)
        loss.backward()
        optimizer.step()
        bs = lbls_b.size(0)
        train_loss += loss.item()*bs
        total      += bs
    train_loss /= total

    # — Validation —
    model.eval()
    val_loss = 0.0; correct=0; total_v=0
    with torch.no_grad():
        for imgs_b, lbls_b in val_loader:
            imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
            out = model(imgs_b)
            l   = criterion(out, lbls_b)
            val_loss += l.item()*lbls_b.size(0)
            correct  += (out.argmax(1)==lbls_b).sum().item()
            total_v  += lbls_b.size(0)
    val_loss /= total_v
    val_acc  = correct/total_v
    elapsed  = time.time() - start_time

    print(f"[Epoch {epoch}/{total_epochs}] phase={phase} "
          f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)")

    # 메트릭 기록
    metrics.append({
        'epoch':     epoch,
        'phase':     phase,
        'train_loss':train_loss,
        'val_loss':  val_loss,
        'val_acc':   val_acc,
        'time_sec':  round(elapsed,2)
    })

    # Early Stopping & Save
    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_model_efficientnet_b0_two_phase.pth')
        print(f"→ New best saved (val_loss={val_loss:.4f})\n")
    else:
        no_improve_epochs += 1
        print(f"  (no improvement for {no_improve_epochs}/{early_stop_patience} epochs)")

    if no_improve_epochs >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# ─────────────────────────────────────────────────────────────
# 6) JSON로 메트릭 저장
# ─────────────────────────────────────────────────────────────
with open('training_metrics_two_phase.json', 'w') as fp:
    json.dump(metrics, fp, indent=2)

print("✅ Done. Metrics saved to training_metrics_two_phase.json")
