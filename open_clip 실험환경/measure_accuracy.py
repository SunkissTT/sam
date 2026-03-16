import torch
import open_clip
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import numpy as np

# 설정
model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'
device = "cpu"

print(f"로드 중: {model_name} ({pretrained})...")
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_name)

# 데이터셋
dataset = CIFAR100(root='./data', train=False, download=True, transform=preprocess)
loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
classes = dataset.classes

# 텍스트 피처 생성
print("텍스트 임베딩 생성 중...")
with torch.no_grad():
    text_inputs = tokenizer([f"a photo of a {c}" for c in classes]).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

correct_1 = 0
total = 0

print("추론 시작 (10,000장)...")
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity.topk(1, dim=-1)
        
        correct_1 += (indices.flatten() == labels).sum().item()
        total += labels.size(0)
        
        if total % 1024 == 0:
            print(f"진행: {total}/10000...")

accuracy = correct_1 / total
print(f"\n최종 결과:")
print(f"Top-1 Accuracy: {accuracy:.4f} ({correct_1}/{total})")

with open('accuracy_result.txt', 'w') as f:
    f.write(f"{accuracy:.4f}")
