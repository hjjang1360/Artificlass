import os
import time
import json                             # ← 추가
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np

class ResNet50Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        in_feats  = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.drop = nn.Dropout(0.2)
        self.fc1  = nn.Linear(in_feats, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(512, 128)
        self.fc3  = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)        # → (batch, 2048)
        x = self.drop(x)
        x = self.relu(self.fc1(x))  # → (batch, 1024)
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)             # → (batch, num_classes)
        return x


class EfficientNetB0Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ImageNet pretrained EfficientNet-B0 불러오기
        backbone = models.efficientnet_b0(pretrained=True)
        in_feats = backbone.classifier[1].in_features
        # 기존 classifier 제거
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # 새 분류기 추가
        self.drop = nn.Dropout(0.2)
        self.fc1  = nn.Linear(in_feats, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(512, 128)
        self.fc3  = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)          # → (batch, in_feats)
        x = self.drop(x)
        x = self.relu(self.fc1(x))    # → (batch, 512)
        x = self.drop(x)
        x = self.relu(self.fc2(x))    # → (batch, 128)
        x = self.drop(x)
        return self.fc3(x)            # → (batch, num_classes)