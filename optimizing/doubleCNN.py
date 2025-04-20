import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import tqdm
import optuna
from sklearn.preprocessing import LabelEncoder
import torchvision.io as io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



# Directories for JSON logs
# LOG_DIR = "./logs"
LOG_DIR = "/home/work/workspace_ai/Artificlass/logs/double"
# LOG_DIR="../logs"
BEST_MODEL_DIR = "/home/work/workspace_ai/Artificlass/weights/double"
# BEST_MODEL_DIR="../weights/trip"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
class CNN_trip(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(256 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 51)
    def forward(self, x):
        x=self.conv2(self.conv1(x))
        x=self.pool(F.relu(x))
        x=self.dropout(x)
        x=self.conv4(self.conv3(x))
        x=self.pool(F.relu(x))
        x=self.dropout(x)
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        # x=self.dropout(x)
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    

class StyleLabelMapper:
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def get_style_name(self, encoded_label):
        return self.label_encoder.inverse_transform([encoded_label])[0]

    def get_all_style_names(self):
        return self.label_encoder.classes_

# Objective function for Optuna that minimizes validation loss.
def objective(trial):
    # Hyperparameter suggestions 
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # 3 types
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop", "Adamax"]) # 4 types
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lr=trial.suggest_categorical("lr",[1e-4, 1e-3, 1e-2, 1e-1]) # 4 types
    epochs = trial.suggest_categorical("epochs", [100]) # 1 types
    patience = trial.suggest_categorical("patience", [10])  # 2 Early stopping patience

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_trip().to(device)

    # Select optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer choice.")

    criterion = nn.CrossEntropyLoss()

    # Data transforms for training and testing
    transform=transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.ToTensor(),
    transforms.Lambda(lambda x: x/ 255.0),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    ######## DATA LOADING ########
    path_to_data ='/home/work/workspace_ai/Artificlass/data_process/data/'
    csv_file=path_to_data+'imagesinfo.csv'
    data=pd.read_csv(csv_file)
    image_path=path_to_data+'images/'+data['filename'].values
    artist_name=data['artist'].values
    style_name=data['style'].values
    title=data['title'].values

    style_encoder = LabelEncoder()
    title_encoder = LabelEncoder()
    artist_encoder = LabelEncoder()
    style_mapper = StyleLabelMapper(style_encoder)
    title_mapper = StyleLabelMapper(title_encoder)
    artist_mapper = StyleLabelMapper(artist_encoder)
    
    num_sample=1000
    
    style_name_encoded = style_encoder.fit_transform(style_name[:num_sample])
    title_encoded = title_encoder.fit_transform(title[:num_sample])
    artist_encoded = artist_encoder.fit_transform(artist_name[:num_sample])
    
    ####### Making image dataset #######
    img_all = torch.empty((0, 3, 256, 256)).to(device)
    for i in range(len(image_path[:num_sample])):
        img_tmp = io.read_image(image_path[i])
        # Apply the transform to the image
        img_tmp = transform(img_tmp)
        img_all=torch.cat((img_all, img_tmp.unsqueeze(0)), 0) if i > 0 else img_tmp.unsqueeze(0)
    
    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(img_all, style_name_encoded, test_size=0.2, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    # Convert the labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_set=torch.utils.data.TensorDataset(x_train, y_train)
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    valid_set=torch.utils.data.TensorDataset(x_valid, y_valid)
    val_loader=torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False)

    test_set=torch.utils.data.TensorDataset(x_test, y_test)
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Logging dictionary to store metrics for each epoch
    trial_metrics = {
        "trial": [trial.number],
        "batch_size": [batch_size],
        "optimizer": [optimizer_name],
        "learning rate": [lr],
        "patience": [patience],
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

        # Report intermediate validation loss to Optuna for pruning
        trial.report(val_loss, epoch)
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
    return best_val_loss

def run_optuna(n_trials=48):
    # Define the search space as a dictionary mapping each hyperparameter
    search_space = {
        "batch_size": [32, 64, 128],
        "optimizer": ["SGD", "Adam", "RMSprop", "Adamax"],
        "lr": [1e-4, 1e-3, 1e-2, 1e-1],
        "epochs": [100],
        "patience": [10]
    }
    
    # Use GridSampler to ensure every unique combination is run without duplication.
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Study statistics:")
    print("  Number of finished trials: ", len(study.trials))
    best_trial = study.best_trial
    print("  Best trial:")
    print("    Value (Best Validation Loss): {:.4f}".format(best_trial.value))
    print("    Params:")
    for key, value in best_trial.params.items():
        print("      {}: {}".format(key, value))

    # Load the best trial's metrics and save the best trial log as before...
    best_trial_metrics_file = os.path.join(LOG_DIR, f"trial_{best_trial.number}.json")
    best_trial_log = {
        "trial_number": best_trial.number,
        "best_val_loss": best_trial.value,
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
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization with JSON Logging (Minimize Val Loss)")
    parser.add_argument('--n_trials', type=int, default=48, help='Number of Optuna trials')
    args = parser.parse_args()
    run_optuna(n_trials=args.n_trials)
