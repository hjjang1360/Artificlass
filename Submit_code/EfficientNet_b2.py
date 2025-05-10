
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import amp

# 1) 설정
torch.backends.cudnn.benchmark = True

root = '/home/hjjang/Artificlass/data_process/data/augmented_images_split'
batch_size = 16
num_workers = 6
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

# 3) 모델 정의
class EfficientNetB2Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b2(pretrained=True)
        feat = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(feat,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,num_classes)
    def forward(self,x):
        x = self.backbone(x)
        x = self.drop(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

model = EfficientNetB2Head(num_classes).to(device)
# PyTorch 2 컴파일 최적화
if hasattr(torch, 'compile'):
    model = torch.compile(model)

# 4) 옵티마이저 & 스케일러
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer=optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
scaler = amp.GradScaler()

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
        with amp.autocast(device_type='cuda'):
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
        torch.save(model.state_dict(), 'best_b2_fixed_SGD.pth')
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
with open('history_b2_fixed_SGD.json','w') as f:
    json.dump(history, f, indent=2)

print('Done')
