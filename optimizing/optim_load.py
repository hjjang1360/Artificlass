import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

class ParquetMapDataset(Dataset):
    def __init__(self, parquet_path, style2idx, transform=None):
        self.style2idx = style2idx
        self.transform = transform

        # Open file & inspect metadata
        print("🔍 Opening Parquet file and reading metadata...")
        pf = pq.ParquetFile(parquet_path)
        num_groups = pf.num_row_groups
        total_rows = pf.metadata.num_rows
        print(f"   • {num_groups} row-groups, ~{total_rows} total rows found")

        imgs_accum = []
        labels_accum = []

        # Read each row-group with a progress print
        for i in range(num_groups):
            print(f"⏳ Loading row-group {i+1}/{num_groups}...")
            rg = pf.read_row_group(i, columns=['image', 'style'], use_threads=True)

            # 1) pull out image blobs as Python objects and stack into a single NumPy array
            imgs_list = rg.column('image').to_pylist()           # list of H×W×C uint8 arrays
            imgs_np   = np.stack(imgs_list, axis=0)              # shape (B, H, W, C)

            # 2) pull styles as a Python list and map to ints
            styles     = rg.column('style').to_pylist()
            labels_np  = np.array([style2idx[s] for s in styles], dtype=np.int64)

            imgs_accum.append(imgs_np)
            labels_accum.append(labels_np)

        # Concatenate everything into single large arrays
        print("🔗 Concatenating all chunks into RAM arrays…")
        self.images = np.concatenate(imgs_accum, axis=0)  # shape (N, H, W, C)
        self.labels = np.concatenate(labels_accum, axis=0)  # shape (N,)
        print(f"✅ Finished loading: {self.images.shape[0]} samples in RAM")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # pull out one sample
        img = self.images[idx].astype(np.float32) / 255.0  # H×W×C float
        img = torch.from_numpy(img).permute(2, 0, 1)       # → C×H×W

        if self.transform:
            img = self.transform(img)

        label = int(self.labels[idx])
        return img, label


# ─── USAGE ─────────────────────────────────────────────────────────

style_list = [
    'Impressionism', 'Realism', 'Romanticism',
    'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)'
]
style2idx = {s: i for i, s in enumerate(style_list)}

norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

ds = ParquetMapDataset(
    parquet_path='/home/work/workspace_ai/Artificlass/data_process/data/top6_styles_merged.parquet',
    style2idx=style2idx,
    transform=norm
)

loader = DataLoader(
    ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

device = 'cuda'
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(style2idx))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    model.train()
    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        out   = model(imgs)
        loss  = criterion(out, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")
