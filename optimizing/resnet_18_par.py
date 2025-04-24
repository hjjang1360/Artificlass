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


import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # 추가

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
            imgs_list = batch.column('image').to_pylist()
            styles_list = batch.column('style').to_pylist()
            imgs_np = np.stack(imgs_list, axis=0).astype(np.uint8)

            for img_arr, style_str in zip(imgs_np, styles_list):
                img_t = torch.from_numpy(img_arr.astype(np.float32) / 255.0)
                if self.transform:
                    img_t = self.transform(img_t)
                label = self.style2idx[style_str]
                yield img_t, label

# 스타일→인덱스 맵
style_list = ['Impressionism', 'Realism', 'Romanticism',
              'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)']
style2idx = {s: i for i, s in enumerate(style_list)}

# Dataset & DataLoader
ds = ParquetIterableDataset(
    parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
    chunk_size=512,
    style2idx=style2idx,
)
loader = DataLoader(
    ds,
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

# 학습 루프에 tqdm 적용
for epoch in range(1, 11):
    model.train()
    epoch_iter = tqdm(loader, desc=f"[Epoch {epoch}/10]", ncols=80)
    for imgs, labels in epoch_iter:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # 프로그레스바에 현재 Loss 표시
        epoch_iter.set_postfix(loss=loss.item())

    print(f"Epoch {epoch} done")
