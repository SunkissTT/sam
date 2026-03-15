import torch
import torch.nn.functional as F
from datasets import load_dataset
import open_clip
from tqdm import tqdm
import os

# 설정
model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'
device = "cpu"

print(f"모델 로드 중: {model_name} ({pretrained})...")
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_name)

# 클래스명 로드 (open_clip 내부 메타데이터 활용)
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
classes = IMAGENET_CLASSNAMES

# 텍스트 임베딩 생성
print("텍스트 임베딩 생성 중...")
with torch.no_grad():
    text_inputs = tokenizer([f"a photo of a {c}" for c in classes]).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

from torchvision.datasets import ImageFolder

# ImageNet-V2 로컬 데이터셋 로드
data_dir = "/home/daejun/shi_2026/data/imagenetv2-matched-frequency-format-val"
print(f"로컬 ImageNet-V2 로드 중: {data_dir}")

class NumericalImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = sorted(os.listdir(directory), key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

try:
    dataset = NumericalImageFolder(root=data_dir, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
except Exception as e:
    print(f"데이터셋 로드 실패: {e}")
    exit(1)

correct_1 = 0
total = 0
max_samples = 10000 # ImageNet-V2 전체 크기

print(f"추론 시작 (총 {max_samples}개)...")
model.eval()
with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T)
        preds = similarity.argmax(dim=-1)
        
        correct_1 += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct_1 / total
print(f"\n최종 결과 (샘플 {total}개 기준):")
print(f"ImageNet Zero-shot Top-1 Accuracy: {accuracy:.4f}")

with open('imagenet_accuracy_result.txt', 'w') as f:
    f.write(f"{accuracy:.4f}")
