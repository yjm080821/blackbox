import os
from PIL import Image
from torchvision import transforms
import timm

# 테스트 데이터 폴더 경로
test_folder = './data/testing_data/'

# 전처리 변환
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 이미지 경로 리스트
test_images = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith('.jpg')]


import torch.nn as nn
import torch

# 모델 로드
model = timm.create_model('inception_resnet_v2', pretrained=True)

# 출력 레이어 수정 (이진 분류)
model.classif = nn.Sequential(
    nn.Linear(model.classif.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 모델 불러오기
model.load_state_dict(torch.load('inception_resnet_v2_smoking_detector.pth'))
model.eval()

def evaluate_images(model, image_paths):
    results = []
    for img_path in image_paths:
        # 이미지 열기
        img = Image.open(img_path)

        # 이미지가 흑백인 경우 RGB로 변환
        if img.mode != "RGB":
            img = img.convert("RGB")

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
with open('test_results.txt', 'w', encoding='utf-8') as f:  # utf-8 인코딩 추가
    for img_path, label in test_results:
        f.write(f"{img_path}: {label}\n")
