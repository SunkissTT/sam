import torch
from torchvision.datasets import Food101
import open_clip
from tqdm import tqdm
import os

def measure_accuracy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 모델 로드
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    preprocess = preprocess_val
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Food-101 데이터셋 로드
    # download=True 옵션으로 필요 시 다운로드 (약 5GB)
    print("Loading Food-101 dataset...")
    dataset = Food101(root="./data_food101", split="test", download=True, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    classes = dataset.classes
    text_inputs = tokenizer([f"a photo of {c}, a type of food." for c in classes]).to(device)

    print("Encoding text features...")
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    print("Starting inference...")
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T)
            preds = similarity.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"\nFood-101 Zero-shot Accuracy: {accuracy:.4f} ({correct}/{total})")

    with open("food101_accuracy_result.txt", "w") as f:
        f.write(f"{accuracy:.4f}")

if __name__ == "__main__":
    measure_accuracy()
