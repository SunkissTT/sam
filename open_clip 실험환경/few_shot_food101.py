import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
import open_clip
import numpy as np
from tqdm import tqdm
import os
import cv2
from PIL import Image

def run_few_shot_food101():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'ViT-B-32'
    pretrained = 'laion2b_s34b_b79k'
    
    # 모델 로드
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    
    # 데이터셋 로드
    print("Loading Food-101 dataset...")
    train_dataset = Food101(root="./data_food101", split="train", download=True, transform=preprocess)
    test_dataset = Food101(root="./data_food101", split="test", download=True, transform=preprocess)
    
    # 16-shot 셈플링
    num_shots = 16
    indices = []
    targets = np.array(train_dataset._labels)
    for i in range(101):
        cls_indices = np.where(targets == i)[0]
        selected = np.random.choice(cls_indices, num_shots, replace=False)
        indices.extend(selected)
    
    few_shot_train = Subset(train_dataset, indices)
    train_loader = DataLoader(few_shot_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Classifier 정의
    classifier = nn.Linear(512, 101).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Linear Probe (16-shot) for Food-101...")
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
    print(f"Food-101 16-shot Accuracy: {accuracy:.4f}")

    with open("food101_fewshot_accuracy.txt", "w") as f:
        f.write(f"{accuracy:.4f}")

if __name__ == "__main__":
    run_few_shot_food101()
