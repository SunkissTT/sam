import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
import open_clip
import numpy as np
from tqdm import tqdm
import os
import cv2
from PIL import Image

def run_few_shot_cifar100():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'ViT-B-32'
    pretrained = 'laion2b_s34b_b79k'
    
    # 모델 로드
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    
    # 데이터셋 로드
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=preprocess)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=preprocess)
    
    # 16-shot 셈플링
    num_shots = 16
    indices = []
    for i in range(100):
        cls_indices = np.where(np.array(train_dataset.targets) == i)[0]
        selected = np.random.choice(cls_indices, num_shots, replace=False)
        indices.extend(selected)
    
    few_shot_train = Subset(train_dataset, indices)
    train_loader = DataLoader(few_shot_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 특징 추출 및 Linear Classifier 학습
    # CLIP Image Encoder 뒤에 Linear Layer 추가
    classifier = nn.Linear(512, 100).to(device) # ViT-B-32 output dim is 512
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Extracting features and training Linear Probe (16-shot)...")
    for epoch in range(20):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
            
            outputs = classifier(features.float())
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 평가
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            outputs = classifier(features.float())
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"CIFAR-100 16-shot Accuracy: {accuracy:.4f}")

    with open("cifar100_fewshot_accuracy.txt", "w") as f:
        f.write(f"{accuracy:.4f}")

if __name__ == "__main__":
    run_few_shot_cifar100()
