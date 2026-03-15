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

# Grad-CAM 클래스
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

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
    
    # Grad-CAM 시각화
    target_layer = model.visual.transformer.resblocks[-1].ln_1
    grad_cam = GradCAM(model, target_layer)
    os.makedirs('results_fewshot_food101', exist_ok=True)
    
    def save_overlap(img_pil, cam, path, title):
        img_cv = cv2.cvtColor(np.array(img_pil.resize((224, 224))), cv2.COLOR_RGB2BGR)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlap = heatmap + np.float32(img_cv) / 255
        overlap = overlap / np.max(overlap)
        canvas = np.zeros((264, 448, 3), dtype=np.uint8)
        canvas[40:, :224] = img_cv
        canvas[40:, 224:] = np.uint8(255 * overlap)
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(path, canvas)

    c_cnt, i_cnt = 0, 0
    test_indices = np.random.permutation(len(test_dataset))
    classes = test_dataset.classes

    print("Generating Grad-CAM for Food-101 Few-shot samples...")
    for idx in test_indices:
        if c_cnt >= 10 and i_cnt >= 10: break
        
        img_pil, label = test_dataset[idx] # Dataset returns (PIL, label) but preprocess applied? 
        # Wait, test_dataset has preprocess. I need the original for display.
        # Let's reload without transform for display.
        orig_dataset = Food101(root="./data_food101", split="test", download=False)
        img_pil_orig, _ = orig_dataset[idx]
        
        img_input = preprocess(img_pil_orig).unsqueeze(0).to(device)
        img_input.requires_grad = True
        
        features = model.encode_image(img_input)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = classifier(features.float())
        pred = logits.argmax(dim=-1).item()
        is_correct = (pred == label)
        
        if (is_correct and c_cnt < 10) or (not is_correct and i_cnt < 10):
            model.zero_grad()
            classifier.zero_grad()
            logits[0, pred].backward()
            
            grads = grad_cam.gradients.detach().cpu().numpy()
            acts = grad_cam.activations.detach().cpu().numpy()
            
            weights = np.mean(grads[0, 1:, :], axis=0)
            cam = np.maximum(0, np.dot(acts[0, 1:, :], weights))
            cam = cam.reshape(7, 7)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            if is_correct:
                save_overlap(img_pil_orig, cam, f'results_fewshot_food101/correct_{c_cnt}.png', f'Correct: {classes[label]}')
                c_cnt += 1
            else:
                save_overlap(img_pil_orig, cam, f'results_fewshot_food101/incorrect_{i_cnt}.png', f'Incorrect: GT={classes[label]}, Pred={classes[pred]}')
                i_cnt += 1

    with open("food101_fewshot_accuracy.txt", "w") as f:
        f.write(f"{accuracy:.4f}")

if __name__ == "__main__":
    run_few_shot_food101()
