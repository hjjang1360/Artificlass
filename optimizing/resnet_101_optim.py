import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, TensorDataset
import tqdm
import optuna
from sklearn.preprocessing import LabelEncoder
import torchvision.io as io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
# from tqdm import tqdm
import time



# Directories for JSON logs
# LOG_DIR = "./logs"
LOG_DIR = "/home/work/workspace_ai/Artificlass/logs/resnet101_added1"
# LOG_DIR="../logs"
BEST_MODEL_DIR = "/home/work/workspace_ai/Artificlass/weights/resnet101_added1"
# BEST_MODEL_DIR="../weights/trip"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
class resnet50_add(nn.Module):
    # def __init__(self, num_classes):
    #     super(resnet50_add, self).__init__()
    #     backbone = models.resnet50(pretrained=False)
    #     in_feats = backbone.fc.in_features
    #     backbone.fc = nn.Identity()
    #     self.backbone = backbone
        
    #     self.drop  = nn.Dropout(p=0.2)
    #     self.fc1   = nn.Linear(in_feats, 1024)
    #     self.relu  = nn.ReLU()
    #     self.fc2   = nn.Linear(1024, num_classes)
    #     # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, len(style2idx))
        

    # def forward(self, x):
    #     # return self.resnet(x)
    #     # x= self.resnet(x)
    #     # x=self.delete(x)
    #     # x = self.drop(x)
    #     # x = self.fc1(x)
    #     # x = self.fc2(self.drop(x))
    #     x = self.backbone(x)           # → (batch, 2048)
    #     x = self.drop(x)
    #     x = self.relu(self.fc1(x))     # → (batch, 1024)
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     return x
    def __init__(self, num_classes=7, dropout_rates=(0.5, 0.4, 0.3, 0.2)):
        super().__init__()
        # 1) backbone: ResNet50 up to the final pooling
        self.backbone = models.resnet101(pretrained=False)
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
    

# Objective function for Optuna that minimizes validation loss.
def objective(trial):
    # Hyperparameter suggestions 
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # 3 types
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"]) # 3 types
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    # lr=trial.suggest_categorical("lr",[1e-4, 1e-3, 1e-2, 1e-1]) # 4 types
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    p_hflip = trial.suggest_uniform("p_hflip", 0.0, 1.0)
    p_crop  = trial.suggest_uniform("p_crop", 0.0, 1.0)
    momentum = trial.suggest_uniform("momentum", 0.5, 0.99)
    
    drop_rate1 = trial.suggest_uniform("drop1", 0.1, 0.7)
    drop_rate2 = trial.suggest_uniform("drop2", 0.1, 0.7)
    drop_rate3 = trial.suggest_uniform("drop3", 0.1, 0.7)
    drop_rate4 = trial.suggest_uniform("drop4", 0.1, 0.7)

    epochs = trial.suggest_categorical("epochs", [100]) # 1 types
    patience = trial.suggest_categorical("patience", [10])  # 2 Early stopping patience

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50_add(num_classes=7, dropout_rates=(drop_rate1,drop_rate2,drop_rate3, drop_rate4))  # 7 classes for top-7 styles
    model.to(device)

    # Select optimizer
    if optimizer_name == 'SGD':
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'Adam':
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # elif optimizer_name == 'Adamax':
    #     optimizer = optim.Adamax(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer choice.")

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
    h5_path = '/home/work/workspace_ai/Artificlass/data_process/data/top7_h5_merged.h5'
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
    # normalize = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=p_hflip),
    #     # transforms.RandomCrop(256, padding=4, p=trial.suggest_uniform("p_crop",0,1)),
    #     transforms.RandomApply(
    #     [transforms.RandomCrop(224, padding=4)],
    #     p=p_crop
    #     ),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std =[0.229, 0.224, 0.225])
    # ])
    # imgs = normalize(imgs)       
    # img = torch.from_numpy(raw).float()
    img = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=p_hflip),
        transforms.ToTensor(),            # (C,H,W)로
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img) 

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
    # batch_size = 64
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    
    best_val_loss = float('inf')
    best_test_accuracy = 0.0
    epochs_no_improve = 0

    # Logging dictionary to store metrics for each epoch
    trial_metrics = {
        "trial": [trial.number],
        "batch_size": [batch_size],
        "optimizer": [optimizer_name],
        "learning rate": [lr],
        "patience": [patience],
        "weight_decay": [weight_decay],
        "p_hflip": [p_hflip],
        "p_crop": [p_crop],
        "momentum": [momentum],
        "drop_rate": [drop_rate1, drop_rate2, drop_rate3, drop_rate4],
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "test_accuracy": []
    }
    print(f"Trial {trial.number} - Batch Size: {batch_size}, Optimizer: {optimizer_name}, Learning Rate: {lr}, Epochs: {epochs}, Patience: {patience}")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm.tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{epochs} - Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Testing phase (for reference)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total

        # Log metrics for this epoch
        trial_metrics["epochs"].append(epoch + 1)
        trial_metrics["train_loss"].append(train_loss)
        trial_metrics["val_loss"].append(val_loss)
        trial_metrics["test_accuracy"].append(test_accuracy)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"best_model_trial_{trial.number}.pth"))
            print(f"Trial {trial.number} - New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"best_model_acc_trial_{trial.number}.pth"))

        # Report intermediate validation loss to Optuna for pruning
        # trial.report(val_loss, epoch)
        trial.report(test_accuracy, epoch)
        if trial.should_prune():
            # Save the trial metrics before pruning
            trial_log_file = os.path.join(LOG_DIR, f"trial_{trial.number}.json")
            with open(trial_log_file, "w") as f:
                json.dump(trial_metrics, f, indent=4)
            raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} for Trial {trial.number}")
            break

        print(f"Trial {trial.number} Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    
    # Save per-trial metrics into a JSON file
    trial_log_file = os.path.join(LOG_DIR, f"trial_{trial.number}.json")
    with open(trial_log_file, "w") as f:
        json.dump(trial_metrics, f, indent=4)

    # Return the best (minimum) validation loss for this trial
    # return best_val_loss
    return best_test_accuracy

def run_optuna(n_trials=48):
    # Define the search space as a dictionary mapping each hyperparameter
    search_space = {
        "batch_size": [32, 64, 128],
        "optimizer": ["SGD", "Adam", "RMSprop", "Adamax"],
        "lr": [1e-4, 1e-3, 1e-2, 1e-1],
        "epochs": [100],
        "patience": [5]
    }
    sampler = optuna.samplers.TPESampler()
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    print("Using sampler:", study.sampler.__class__.__name__)
    study.optimize(objective, n_trials=n_trials)
    
    # Use GridSampler to ensure every unique combination is run without duplication.
    # sampler = optuna.samplers.GridSampler(search_space)
    # study = optuna.create_study(direction="minimize", sampler=sampler)
    # study.optimize(objective, n_trials=n_trials)

    print("Study statistics:")
    print("Using sampler:", study.sampler.__class__.__name__)
    print("  Number of finished trials: ", len(study.trials))
    best_trial = study.best_trial
    print("  Best trial:")
    # print("    Value (Best Validation Loss): {:.4f}".format(best_trial.value))
    print(f"  Best trial #{best_trial.number} — test_accuracy: {best_trial.value:.4f}%")
    print("    Params:")
    # for key, value in best_trial.params.items():
    #     print("      {}: {}".format(key, value))
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # Load the best trial's metrics and save the best trial log as before...
    best_trial_metrics_file = os.path.join(LOG_DIR, f"trial_{best_trial.number}.json")
    best_trial_log = {
        "trial_number": best_trial.number,
        # "best_val_loss": best_trial.value,
        best_test_accuracy: best_trial.value,
        "params": best_trial.params,
    }
    if os.path.exists(best_trial_metrics_file):
        with open(best_trial_metrics_file, "r") as f:
            best_trial_metrics = json.load(f)
        best_trial_log["metrics"] = best_trial_metrics

    best_log_path = os.path.join(BEST_MODEL_DIR, "best_model_trial.json")
    with open(best_log_path, "w") as f:
        json.dump(best_trial_log, f, indent=4)
    print(f"Best trial log saved to: {best_log_path}")

if __name__ == '__main__':
    # You can pass the number of trials via command-line arguments if needed.
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization with JSON Logging (Maximize Test Accuracy)")
    parser.add_argument('--n_trials', type=int, default=48, help='Number of Optuna trials')
    args = parser.parse_args()
    run_optuna(n_trials=args.n_trials)
