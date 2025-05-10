
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
# from torch import amp
from torch.cuda.amp import autocast, GradScaler

# 1) 설정
torch.backends.cudnn.benchmark = True

root = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split_v3'
batch_size = 64
num_workers = 4
num_classes = 7
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2) 데이터 로더
train_tf = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(512, scale=(0.8,1.0)),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize(540),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)
train_ds = datasets.ImageFolder(os.path.join(root,'train'), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(root,'val'),   transform=val_tf)
test_ds  = datasets.ImageFolder(os.path.join(root,'test'),  transform=val_tf)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

# class MobileNetV3SmallHead(nn.Module):
#     def __init__(self, num_classes, dropout=0.2):
#         super().__init__()
#         # 1) 사전학습된 MobileNetV3-Small 로드
#         backbone = models.mobilenet_v3_small(pretrained=True)
#         # 2) 원래 classifier 부분 제거
#         #    (backbone.classifier: [Dropout, Linear(in_feats → 1000)])
#         in_feats = backbone.classifier[1].in_features
#         backbone.classifier = nn.Identity()
#         self.backbone = backbone

#         # 3) 새 분류기 헤드 정의
#         self.head = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(in_feats, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.backbone(x)  # → (batch, in_feats)
#         return self.head(x)   # → (batch, num_classes)

class MobileNetV3SmallHead(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        # 1) 사전학습된 MobileNetV3-Small 로드
        backbone = models.mobilenet_v3_small(pretrained=True)
        # 2) 새 헤드 입력 차원 추출 (첫 번째 Linear 레이어)
        in_feats = backbone.classifier[0].in_features
        # 3) 기존 classifier 지우기
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # 4) 새 분류기 헤드 정의
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)     # -> (batch, in_feats)
        return self.head(x)      # -> (batch, num_classes)

model = MobileNetV3SmallHead(num_classes=7, dropout=0.2).to(device)
# PyTorch 2 컴파일 최적화
# if hasattr(torch, 'compile'):
#     model = torch.compile(model)

# 4) 옵티마이저 & 스케일러
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# 5) 학습 루프 (Early Stopping 포함)
history = []
epochs = 30
best_val_loss = float('inf')
no_improve_epochs = 0
patience = 5

for epoch in range(1, epochs+1):
    t0 = time.time()
    # Train
    model.train()
    train_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f'Epoch[{epoch}/{epochs}] Train', leave=False):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            out = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc=f'Epoch[{epoch}/{epochs}] Val', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            out = model(imgs)
            val_loss += criterion(out, lbls).item() * imgs.size(0)
            correct += (out.argmax(1)==lbls).sum().item()
    val_loss /= len(val_ds)
    val_acc = correct / len(val_ds)

    # Test Accuracy
    test_correct = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            out = model(imgs)
            test_correct += (out.argmax(1)==lbls).sum().item()
    test_acc = test_correct / len(test_ds)

    # Early Stopping & Model Save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), '/home/work/workspace_ai/Artificlass/real_use_code/light_log/best_b2_fixed.pth')
        print(f'→ New best model saved (val_loss={val_loss:.4f})')
    else:
        no_improve_epochs += 1
        print(f'(No improvement for {no_improve_epochs}/{patience} epochs)')
        if no_improve_epochs >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    # Log
    elapsed = time.time() - t0
    print(f'Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} ' 
          f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} ' 
          f'test_acc={test_acc:.4f} ({elapsed:.1f}s)')
    history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss':   val_loss,
        'val_acc':    val_acc,
        'test_acc':   test_acc,
        'time':       elapsed
    })

# 6) 기록 저장
with open('/home/work/workspace_ai/Artificlass/real_use_code/light_log/history_b2_fixed.json','w') as f:
    json.dump(history, f, indent=2)

print('Done')
