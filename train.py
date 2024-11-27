import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import timm

# ----------1. 데이터셋 전처리----------
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # RGB 변환
    transforms.RandomResizedCrop(299, scale=(0.9, 1.0)),  # 스케일 수정
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, fill=(0, 0, 0)),  # 회전 각도 제한
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 변화 폭 제한
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------2. 데이터셋 로드----------
class SafeImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Skipping corrupted image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # 다음 샘플로 대체

train_dataset_raw = datasets.ImageFolder(root='./data/training_data')
val_dataset_raw = datasets.ImageFolder(root='./data/validation_data')

train_dataset = SafeImageDataset(train_dataset_raw, transform=train_transform)
val_dataset = SafeImageDataset(val_dataset_raw, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

# ----------3. 모델 정의----------
model = timm.create_model('inception_resnet_v2', pretrained=True)

# 출력 레이어 수정 (이진 분류)
model.classif = nn.Sequential(
    nn.Linear(model.classif.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout 추가
    nn.Linear(512, 1)
)

# ----------4. 학습 설정----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-3)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=10)  # T_max는 주기(epoch)

# ----------5. Early Stopping----------
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping()

# ----------6. 학습 루프----------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=20):
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item()

                preds = (outputs.view(-1) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

        print(f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {correct / total:.4f}")

        scheduler.step()
        if early_stopping(val_loss / len(val_loader)):
            print("Early stopping triggered.")
            break

    # Plot training curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# ----------7. 학습 실행----------
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping)

# ----------8. 모델 저장----------
torch.save(model.state_dict(), "inception_resnet_v2_smoking_detector.pth")
