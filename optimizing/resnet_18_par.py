# import pyarrow.parquet as pq
# import torch
# from torch.utils.data import IterableDataset, DataLoader
# from torchvision import transforms, models
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # 1) IterableDataset 정의
# class ParquetIterableDataset(IterableDataset):
#     def __init__(self, parquet_path, chunk_size=512, style2idx=None, transform=None):
#         self.parquet_path = parquet_path
#         self.chunk_size = chunk_size
#         self.style2idx = style2idx or {}
#         self.transform = transform

#     def __iter__(self):
#         pf = pq.ParquetFile(self.parquet_path)
#         for batch in pf.iter_batches(batch_size=self.chunk_size,
#                                      columns=['image','style'],
#                                      use_threads=True):
#             # ★ Python 리스트로 넘어온 image 컬럼을 NumPy 배열로 변환
#             imgs_list = batch.column('image').to_pylist()   # list of (3,256,256) lists
#             styles_list = batch.column('style').to_pylist() # list of style strings

#             # 리스트 모양 그대로 쌓아서 (N,3,256,256) NumPy 배열로
#             imgs_np = np.stack(imgs_list, axis=0).astype(np.uint8)

#             for img_arr, style_str in zip(imgs_np, styles_list):
#                 # [0,255] → [0,1] float tensor
#                 img_t = torch.from_numpy(img_arr.astype(np.float32) / 255.0)
                
#                 if self.transform:
#                     img_t = self.transform(img_t)
                
#                 label = self.style2idx[style_str]
#                 yield img_t, label


# # 2) 스타일→인덱스 맵 생성 (미리 CSV나 메타데이터에서)
# # style_list = ['Impressionism','Cubism', ...]  # 실제 6개 스타일 이름
# style_list=['Impressionism', 'Realism', 'Romanticism', 'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)']
# style2idx = {s:i for i,s in enumerate(style_list)}

# # 3) Dataset·DataLoader 인스턴스
# ds = ParquetIterableDataset(
#     # parquet_path='/path/to/top6_styles_merged.parquet',
#     parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
#     chunk_size=512,
#     style2idx=style2idx,
# )
# loader = DataLoader(
#     ds,
#     batch_size=64,       # GPU 메모리에 맞춰 조정
#     num_workers=4,       # I/O 병렬
#     prefetch_factor=2,   # 각 워커가 미리 갖고 올 배치 수
# )

# # 4) ResNet-18 설정
# device = 'cuda'
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, len(style2idx))
# model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

# # 5) 학습 루프 (예시)
# for epoch in range(10):
#     model.train()
#     for imgs, labels in loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         out = model(imgs)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch} done')


# import pyarrow.parquet as pq
# import torch
# from torch.utils.data import IterableDataset, DataLoader
# from torchvision import transforms, models
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm  # 추가

# class ParquetIterableDataset(IterableDataset):
#     def __init__(self, parquet_path, chunk_size=512, style2idx=None, transform=None):
#         self.parquet_path = parquet_path
#         self.chunk_size = chunk_size
#         self.style2idx = style2idx or {}
#         self.transform = transform

#     def __iter__(self):
#         pf = pq.ParquetFile(self.parquet_path)
#         for batch in pf.iter_batches(batch_size=self.chunk_size,
#                                      columns=['image','style'],
#                                      use_threads=True):
#             imgs_list = batch.column('image').to_pylist()
#             styles_list = batch.column('style').to_pylist()
#             imgs_np = np.stack(imgs_list, axis=0).astype(np.uint8)

#             for img_arr, style_str in zip(imgs_np, styles_list):
#                 img_t = torch.from_numpy(img_arr.astype(np.float32) / 255.0)
#                 if self.transform:
#                     img_t = self.transform(img_t)
#                 label = self.style2idx[style_str]
#                 yield img_t, label

# # 스타일→인덱스 맵
# style_list = ['Impressionism', 'Realism', 'Romanticism',
#               'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)']
# style2idx = {s: i for i, s in enumerate(style_list)}

# # Dataset & DataLoader
# ds = ParquetIterableDataset(
#     parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
#     chunk_size=512,
#     style2idx=style2idx,
# )
# loader = DataLoader(
#     ds,
#     batch_size=64,
#     num_workers=4,
#     prefetch_factor=2,
# )

# # 모델/최적화기/손실함수 설정
# device = 'cuda'
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, len(style2idx))
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

# # 학습 루프에 tqdm 적용
# for epoch in range(1, 11):
#     model.train()
#     epoch_iter = tqdm(loader, desc=f"[Epoch {epoch}/10]", ncols=80)
#     for imgs, labels in epoch_iter:
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         out = model(imgs)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()

#         # 프로그레스바에 현재 Loss 표시
#         epoch_iter.set_postfix(loss=loss.item())

#     print(f"Epoch {epoch} done")

# import pyarrow.parquet as pq
# import torch
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision import transforms, models
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm

# # 1) Map-style Dataset 정의: 한 번에 전체를 메모리에 로드
# class ParquetDataset(Dataset):
#     def __init__(self, parquet_path, style2idx, transform=None, max_samples=None):
#         # Parquet 파일 전체 읽기
#         table = pq.read_table(parquet_path, columns=['image', 'style'], use_threads=True)
#         # read_opts = pq.ReadOptions(max_rows=20000) if max_samples else None
#         # table = pq.read_table(
#         #     parquet_path,
#         #     columns=['image','style'],
#         #     read_options=read_opts
#         # )
#         imgs_list = table.column('image').to_pylist()
#         styles_list = table.column('style').to_pylist()

#         # 최대 샘플 수 제한
#         if max_samples is not None:
#             imgs_list = imgs_list[:max_samples]
#             styles_list = styles_list[:max_samples]

#         # NumPy → Tensor 변환
#         # (N, H, W, C) 형태라면 permute(0,3,1,2) 해줘야 합니다.
#         imgs_np = np.stack(imgs_list, axis=0).astype(np.float32) / 255.0
#         # 예시: imgs_np.shape == (N, H, W, C)
#         imgs_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2)

#         labels = [style2idx[s] for s in styles_list]
#         labels_t = torch.tensor(labels, dtype=torch.long)

#         # optional transform (Normalize 등)
#         if transform is not None:
#             imgs_t = transform(imgs_t)

#         self.images = imgs_t
#         self.labels = labels_t

#     def __len__(self):
#         return len(self.labels) 

#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]

# # 2) 스타일→인덱스 맵
# style_list = ['Impressionism', 'Realism', 'Romanticism',
#               'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)']
# style2idx = {s: i for i, s in enumerate(style_list)}

# # 3) Dataset & DataLoader 생성
# transform = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std =[0.229, 0.224, 0.225]
# )

# ds = ParquetDataset(
#     parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
#     style2idx=style2idx,
#     transform=transform,
#     max_samples=1000
# )

# loader = DataLoader(
#     ds,
#     batch_size=64,
#     shuffle=True,       # Map-style이므로 이제 shuffle 가능
#     num_workers=4,
#     prefetch_factor=2,
# )

# # 4) 모델/최적화/학습 루프는 그대로
# device = 'cuda'
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, len(style2idx))
# model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(1, 11):
#     model.train()
#     epoch_iter = tqdm(loader, desc=f"[Epoch {epoch}/10]", ncols=80)
#     for imgs, labels in epoch_iter:
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         out = model(imgs)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_iter.set_postfix(loss=loss.item())
#     print(f"Epoch {epoch} done")
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gc  # ← 추가

class ParquetIterableDataset(IterableDataset):
    def __init__(self, parquet_path, chunk_size=512, style2idx=None, transform=None):
        self.parquet_path = parquet_path
        self.chunk_size = chunk_size
        self.style2idx = style2idx or {}
        self.transform = transform

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        for batch in pf.iter_batches(batch_size=self.chunk_size,
                                     columns=['image','style'],
                                     use_threads=True):
            imgs_list   = batch.column('image').to_pylist()
            styles_list = batch.column('style').to_pylist()
            imgs_np     = np.stack(imgs_list, axis=0).astype(np.uint8)

            for img_arr, style_str in zip(imgs_np, styles_list):
                img_t = torch.from_numpy(img_arr.astype(np.float32) / 255.0)
                if self.transform:
                    img_t = self.transform(img_t)
                label = self.style2idx[style_str]
                yield img_t, label

            # ── 이 배치 처리 끝나면 로컬 참조들 삭제 ─────────
            del imgs_list, styles_list, imgs_np, batch
            gc.collect()

        # generator가 완전히 끝나면 ParquetFile 참조도 해제
        del pf
        gc.collect()

# 스타일→인덱스 맵
style_list = ['Impressionism', 'Realism', 'Romanticism',
              'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)']
style2idx = {s: i for i, s in enumerate(style_list)}

# transform 정의
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)

# --- Dataset & DataLoader 설정 ---
train_ds = ParquetIterableDataset(
    parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
    chunk_size=512,
    style2idx=style2idx,
    transform=transform
)
train_loader = DataLoader(
    train_ds,
    batch_size=64,
    num_workers=4,
    prefetch_factor=2,
)

val_ds = ParquetIterableDataset(
    parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
    chunk_size=512,
    style2idx=style2idx,
    transform=transform
)
val_loader = DataLoader(
    val_ds,
    batch_size=64,
    num_workers=4,
    prefetch_factor=2,
)

# 모델/최적화기/손실함수 설정
device = 'cuda'
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(style2idx))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
num_epoch=50
best_val_loss = float('inf')
for epoch in range(1, num_epoch+1):
    # ── training ─────────────────────────────────────
    model.train()
    for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch}/10] train", ncols=80):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        # 배치 후에 GPU 캐시 좀 비워 주기 (필요시)
        del imgs, labels
        torch.cuda.empty_cache()

    # ── validation ────────────────────────────────────
    model.eval()
    val_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item() * labels.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
            del imgs, labels, out
            torch.cuda.empty_cache()

    val_loss /= total
    val_acc = correct / total
    print(f"[Epoch {epoch}/10] val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # ── best 모델 저장 ─────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/home/work/workspace_ai/Artificlass/optimizing/best_model.pth')
        print(f"▶ saved new best (val_loss={val_loss:.4f})")

    # ── epoch 끝날 때 한번 더 메모리 정리 ───────────────
    gc.collect()
    torch.cuda.empty_cache()

