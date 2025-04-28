import h5py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, TensorDataset
# import tqdm
import optuna
from sklearn.preprocessing import LabelEncoder
import torchvision.io as io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
# from tqdm import tqdm
import time
start_load = time.time()

# 1) 스타일→인덱스 매핑
style_list = [
    'Impressionism',
    'Realism',
    'Romanticism',
    'Expressionism',
    'Post-Impressionism',
    'Art Nouveau (Modern)',
    'Baroque'                # If top 7
]
style2idx = {s: i for i, s in enumerate(style_list)}

# 2) HDF5 전체 로드
# h5_path = '/home/work/workspace_ai/Artificlass/data_process/data/top7_h5_merged.h5'
h5_path = '/home/work/workspace_ai/Artificlass/data_process/data/top7_h5_512.h5'
# h5_path='/home/hjjang/Artificlass/data_process/data/top7_h5_merged.h5'
with h5py.File(h5_path, 'r') as f:
    # (N, 3, 256, 256) uint8 → float32 [0,1]
    imgs_np = f['images'][:] .astype(np.float32) / 255.0
    # 문자열 array of bytes or str
    styles_h5 = f['style'][:]  

# 3) NumPy → Torch Tensor
# 이미 (N, C, H, W) 이므로 permute 불필요
imgs = torch.from_numpy(imgs_np)
# 문자열 → 인덱스
# if styles_h5 is bytes, decode first
styles = [s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else s 
          for s in styles_h5]
labels = torch.tensor([style2idx[s] for s in styles], dtype=torch.long)

print(f">> Loaded images: {imgs.shape}, labels: {labels.shape}, load time: {time.time() - start_load:.1f}s")

# 4) 채널 정규화
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)
# transform=transforms.Compose([
#         transforms.RandomCrop(32,padding=4),
#         transforms.RandomHorizontalFlip(),
#         # transforms.ToTensor(),
#         transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std =[0.229, 0.224, 0.225]) # ImageNet의 avg, std. https://pytorch.org/vision/stable/models.html
#         # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#     ])
imgs = normalize(imgs)
# imgs= transform(imgs)
    # 메모리 정리
    

# 5) Dataset → train/val/test 분할
full_ds = TensorDataset(imgs, labels)
n       = len(full_ds)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
n_test  = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_ds,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
print(f">> Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# 6) DataLoader 생성
batch_size = 64
loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

# 7) 모델·옵티마이저·손실함수 설정
device    = 'cuda'
class resnet50_add(nn.Module):
    def __init__(self, num_classes=7, dropout_rates=(0.5, 0.4, 0.3, 0.2)):
        super().__init__()
        # 1) backbone: ResNet50 up to the final pooling
        self.backbone = models.resnet50(pretrained=True)
        # remove the default fc
        self.backbone.fc = nn.Identity()
        
        # 2) our GlobalAveragePooling2D
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3) classifier head
        layers = []
        in_features = 2048
        hidden_sizes = [1024, 512, 256, 128]
        
        for i, out_features in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rates[i]))
            in_features = out_features
        
        # final output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # 4) weight initialization (He / Kaiming uniform for linears)
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
        # backbone returns feature maps: (B, 2048, H', W')
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # global average pool
        x = self.global_pool(x)         # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)         # (B, 2048)
        
        # classifier
        logits = self.classifier(x)     # (B, num_classes)
        return F.softmax(logits, dim=1)
    
# model     = models.resnet50(pretrained=False)
# model.fc  = nn.Linear(model.fc.in_features, len(style2idx))
model=resnet50_add(num_classes=len(style2idx))
model.to(device)

optimizer = optim.NAdam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 8) 학습 + 검증 루프 (최저 val_loss 시 저장)
best_val_loss = float('inf')
num_epochs   = 50
print(f">> Starting training for {num_epochs} epochs", flush=True)

for epoch in range(1, num_epochs+1):
    # — Training —
    model.train()
    train_loss= total=0
    for imgs_b, lbls_b in tqdm(train_loader,
                                desc=f"[Epoch {epoch}/{num_epochs}] train",
                                ncols=80):
        imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs_b), lbls_b)
        total += lbls_b.size(0)
        train_loss += loss.item() * lbls_b.size(0)
        loss.backward()
        optimizer.step()
    train_loss /= total
    print(f"[Epoch {epoch}/{num_epochs}] train_loss: {train_loss:.4f}", end=' ')

    # — Validation —
    model.eval()
    val_loss = correct = total = 0
    correct_t = total_t = 0
    with torch.no_grad():
        for imgs_b, lbls_b in val_loader:
            imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
            out = model(imgs_b)
            val_loss += criterion(out, lbls_b).item() * lbls_b.size(0)
            correct  += (out.argmax(1) == lbls_b).sum().item()
            total    += lbls_b.size(0)
        for imgs_c, lbls_c in test_loader:
            imgs_c, lbls_c = imgs_c.to(device), lbls_c.to(device)
            out = model(imgs_c)
            correct_t += (out.argmax(1) == lbls_c).sum().item()
            total_t   += lbls_c.size(0)

    val_loss /= total
    val_acc  = correct / total
    test_acc = correct_t / total_t
    print(f"[Epoch {epoch}/{num_epochs}] val_loss: {val_loss:.4f}, "
          f"val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_under_h5_top7_pre.pth')
        print(f"▶ New best model saved (val_loss={best_val_loss:.4f})\n")