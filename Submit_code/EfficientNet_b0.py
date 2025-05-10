import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# cuDNN 설정: 고정 입력 크기에 최적화 알고리즘 캐시 사용
# ─────────────────────────────────────────────────────────────
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
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(540),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────────────────────
# 2) ImageFolder 로더 (이미 split된 폴더 구조 사용)
# ─────────────────────────────────────────────────────────────
train_root = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/train'
val_root   = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/val'
test_root  = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/test'

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
# 3) 모델 정의: EfficientNet-B0 + custom head
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
num_classes = len(train_ds.classes)
model = EfficientNetB0Head(num_classes).to(device)

# ─────────────────────────────────────────────────────────────
# 4) Fine-tuning 설정: 단계적 unfreeze + discriminative LR + OneCycle
# ─────────────────────────────────────────────────────────────
freeze_epochs = 5

# 1) backbone freeze → head만 학습
for p in model.backbone.parameters():
    p.requires_grad = False
head_params = list(model.fc1.parameters()) + \
              list(model.fc2.parameters()) + \
              list(model.fc3.parameters())

optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-2)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=freeze_epochs
)

criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# ─────────────────────────────────────────────────────────────
# 5) 학습 + Early Stopping
# ─────────────────────────────────────────────────────────────
best_val_loss     = float('inf')
no_improve_epochs = 0
early_stop_patience = 5
metrics = []
num_epochs = 50

print(f"Starting training for {num_epochs} epochs on device={device}")

for epoch in range(1, num_epochs+1):
    start_time = time.time()

    # (A) freeze_epochs 지나면 backbone unfreeze + optimizer 재설정
    if epoch == freeze_epochs + 1:
        for p in model.backbone.parameters():
            p.requires_grad = True

        optimizer = optim.AdamW([
            {'params': list(model.backbone.parameters()), 'lr': 1e-5},
            {'params': head_params,                       'lr': 1e-4},
        ], weight_decay=1e-2)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=[1e-5, 1e-4],
            steps_per_epoch=len(train_loader),
            epochs=num_epochs - freeze_epochs
        )

    # (B) Training
    model.train()
    train_loss = 0.0
    total = 0
    for imgs, lbls in tqdm(train_loader, desc=f"[Epoch {epoch}] train", ncols=80):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        with autocast():
            out  = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = lbls.size(0)
        train_loss += loss.item() * bs
        total += bs
    train_loss /= total

    # (C) Validation & Test
    model.eval()
    val_loss = 0.0; correct = 0; total_v = 0
    correct_t = 0; total_t = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            l   = criterion(out, lbls)
            val_loss += l.item() * lbls.size(0)
            correct  += (out.argmax(1)==lbls).sum().item()
            total_v  += lbls.size(0)

        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            correct_t += (out.argmax(1)==lbls).sum().item()
            total_t   += lbls.size(0)

    val_loss /= total_v
    val_acc = correct / total_v
    test_acc = correct_t / total_t
    elapsed = time.time() - start_time

    print(f"[Epoch {epoch}/{num_epochs}] "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
          f"test_acc={test_acc:.4f}  ({elapsed:.1f}s)")

    # 기록 저장
    metrics.append({
        'epoch':     epoch,
        'train_loss':train_loss,
        'val_loss':  val_loss,
        'val_acc':   val_acc,
        'test_acc':  test_acc,
        'time_sec':  round(elapsed,2)
    })

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_model_efficientnet_b0_finetuned.pth')
        print(f"→ New best model saved (val_loss={best_val_loss:.4f})\n")
    else:
        no_improve_epochs += 1
        print(f"  (no improvement for {no_improve_epochs}/{early_stop_patience} epochs)")

    if no_improve_epochs >= early_stop_patience:
        print(f"Early stopping triggered after epoch {epoch}")
        break

# ─────────────────────────────────────────────────────────────
# 6) 메트릭 JSON으로 저장
# ─────────────────────────────────────────────────────────────
with open('training_metrics_efficientnet_b0_finetuned.json', 'w') as fp:
    json.dump(metrics, fp, indent=2)

print("✅ Done. Metrics saved to training_metrics_efficientnet_b0_finetuned.json")
