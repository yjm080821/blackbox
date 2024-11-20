#----------1. 데이터셋 로드 및 전처리----------
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 모델 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root='data/training_data', transform=transform)
val_dataset = datasets.ImageFolder(root='data/validation_data', transform=transform)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 클래스 확인 (정상, 비정상 매핑)
print(train_dataset.classes)  # ['정상', '비정상']

from torchvision.models import inception_v3
import torch.nn as nn
import torch

# 모델 로드
model = inception_v3(pretrained=True, aux_logits=True)

# 출력 레이어 수정 (이진 분류)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),  # 이진 분류
    nn.Sigmoid()  # 확률 값 출력
)

#----------2. 모델 구성 및 학습----------

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import torch.optim as optim

criterion = nn.BCELoss()  # 이진 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

#----------3. 학습 루프----------
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device).float()

            # Forward Pass
            outputs = model(inputs)  # InceptionOutputs 반환
            logits = outputs[0]  # logits만 추출
            logits = logits.squeeze(-1)  # 마지막 차원 제거 (batch_size, 1) -> (batch_size,)
            loss = criterion(logits, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loader_tqdm.set_postfix({"Loss": loss.item()})

        # 검증 모드
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                logits = outputs[0]  # logits만 추출
                logits = logits.squeeze(-1)  # 마지막 차원 제거
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (logits > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_loader_tqdm.set_postfix({"Loss": loss.item()})

        val_accuracy = correct / total
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
#----------4. 모델 저장----------
torch.save(model.state_dict(), 'smoking_detector.pth')
#----------5. 모델 예측----------
import os
from PIL import Image
from torchvision import transforms

# 테스트 데이터 폴더 경로
test_folder = 'testing_data/'

# 전처리 변환
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 이미지 경로 리스트
test_images = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith('.jpg')]

# 모델 불러오기
model.load_state_dict(torch.load('smoking_detector.pth'))
model.eval()

def evaluate_images(model, image_paths):
    results = []
    for img_path in image_paths:
        # 이미지 열기
        img = Image.open(img_path)
        
        # 전처리
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # 모델 예측
        with torch.no_grad():
            output = model(input_tensor).item()
        
        # 결과 저장
        label = "흡연" if output > 0.5 else "비흡연"
        results.append((img_path, label))
    return results

test_results = evaluate_images(model, test_images)

# 결과 출력
for img_path, label in test_results[:10]:  # 상위 10개만 출력
    print(f"{img_path}: {label}")

# 결과 파일 저장 (선택)
with open('test_results.txt', 'w') as f:
    for img_path, label in test_results:
        f.write(f"{img_path}: {label}\n")
