import pyarrow.parquet as pq
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time 

start_load=time.time()
# 1) 스타일→인덱스 매핑
style_list = [
    'Impressionism',
    'Realism',
    'Romanticism',
    'Expressionism',
    'Post-Impressionism',
    'Art Nouveau (Modern)'
]
style2idx = {s: i for i, s in enumerate(style_list)}

# 2) Parquet 전체 배치 단위 로드 (to_pylist + np.stack)
imgs_chunks = []
labels_chunks = []
pf = pq.ParquetFile(
    '/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_under_merged.parquet'
)
for i,batch in enumerate(pf.iter_batches(batch_size=1024,
                             columns=['image','style'],
                             use_threads=True)):
    # batch_size=5000
    # 이미지: list of (H,W,C) lists → NumPy → Torch
    imgs_np = np.stack(batch.column('image').to_pylist(), axis=0).astype(np.float32) / 255.0
    # img_t   = torch.from_numpy(imgs_np).permute(0, 3, 1, 2)  # (N, C, H, W)
    img_t=torch.from_numpy(imgs_np)
    imgs_chunks.append(img_t)

    # 레이블: list of style strings → 인덱스 → Tensor
    styles = batch.column('style').to_pylist()
    lbl_t  = torch.tensor([style2idx[s] for s in styles], dtype=torch.long)
    labels_chunks.append(lbl_t)

    print(f"  Batch {i:03d}: loaded {img_t.shape[0]} samples, "
          f"chunk tensor shape={img_t.shape}", flush=True)
    # if i==1:
    #     break

# 3) 청크 결합
imgs   = torch.cat(imgs_chunks,   dim=0)  # (total_N,3,256,256)
labels = torch.cat(labels_chunks, dim=0)  # (total_N,)
print(f">> Finished loading all batches: total images={imgs.shape}, labels={labels.shape}", flush=True)


# 4) 채널 정규화
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)
imgs = normalize(imgs)

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
model     = models.resnet18(pretrained=False)
model.fc  = nn.Linear(model.fc.in_features, len(style2idx))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 8) 학습 + 검증 루프 (최저 val_loss 시 저장)
best_val_loss = float('inf')
num_epochs   = 50
print(f">> Starting training for {num_epochs} epochs", flush=True)

for epoch in range(1, num_epochs+1):
    # — Training —
    model.train()
    for imgs_b, lbls_b in tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}]", ncols=80):
        imgs_b, lbls_b = imgs_b.to(device), lbls_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs_b), lbls_b)
        loss.backward()
        optimizer.step()

    # — Validation —
    model.eval()
    val_loss = correct = total = 0
    correct_t=0
    total_t=0
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
            correct_t  += (out.argmax(1) == lbls_c).sum().item()
            total_t    += lbls_c.size(0)

    val_loss /= total
    val_acc  = correct / total
    test_acc=correct_t/total_t
    print(f"[Epoch {epoch}/{num_epochs}] val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_under.pth')
        print(f"▶ New best model saved (val_loss={best_val_loss:.4f})\n")
