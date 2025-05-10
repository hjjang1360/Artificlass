import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm

# 속도 최적화
torch.backends.cudnn.benchmark = True  # CuDNN autotuner 사용

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=8):
    # 강화된 증강파이프라인 (RandAugment)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(540),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=val_tf)
    test_ds  = datasets.ImageFolder(root=test_dir,  transform=val_tf)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

class StyleClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=7, dropout=0.2):
        super().__init__()
        # timm 백본 불러오기 (checkpointing 지원 모델이면 활성화)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=dropout)
        try:
            self.backbone.set_grad_checkpointing()
        except AttributeError:
            pass

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc=f"Epoch[{epoch}/{total_epochs}] Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, phase="Val", epoch=None, total_epochs=None):
    model.eval()
    loss = 0.0
    correct = 0
    for imgs, labels in tqdm(loader, desc=f"Epoch[{epoch}/{total_epochs}] {phase}", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            loss += criterion(outputs, labels).item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return loss/len(loader.dataset), correct/len(loader.dataset)


def main():
    # 설정
    train_dir = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/train'
    val_dir   = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/val'
    test_dir  = '/home/work/workspace_ai/Artificlass/data_process/data/augmented_images_split/test'
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 7
    epochs = 30
    batch_size = 12  # 배치 사이즈 증가
    lr = 3e-4

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir, val_dir, test_dir, batch_size=batch_size, num_workers=6)

    model = StyleClassifier('efficientnet_b4', num_classes=num_classes, dropout=0.3).to(device)
    # PyTorch 2.0 컴파일
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    # 옵티마이저 & 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 손실함수: 레이블 스무딩
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, epochs+1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, epochs)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "Val", epoch, epochs)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, "Test", epoch, epochs)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | Time: {elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'time': round(elapsed,2)
        })

        # 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/home/work/workspace_ai/Artificlass/real_use_code/Log/b4_eff_best.pth')
            print('→ Saved best model')

    # 기록 저장
    with open('/home/work/workspace_ai/Artificlass/real_use_code/Log/b4_eff_best.json', 'w') as f:
        json.dump(history, f, indent=2)

    print('Training complete.')

if __name__ == '__main__':
    main()
