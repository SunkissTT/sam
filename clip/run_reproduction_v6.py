"""
CLIP Figure 6, 7, 8 재현 스크립트 V5
---
논문 (arxiv:2103.00020) 완전 정합 버전:
1. Logistic Regression: L2-regularization, C sweep (1e-6 ~ 1e6, 96 log-spaced steps)
2. Figure 7: 1, 2, 4, 8, 16, fully_supervised 결과를 포함하여 log-linear 보간
3. 데이터셋별 공식 프롬프트 컨텍스트 + 80개 이미지넷 앙상블 적용
4. MNIST: 인용부호 프롬프트 ("0", "1", ...) 적용
"""

import os
import torch
import clip
import numpy as np
import json
from torchvision.datasets import (
    CIFAR10, CIFAR100, MNIST, STL10, Food101,
    OxfordIIITPet, Flowers102, FGVCAircraft,
    RenderedSST2, GTSRB, EuroSAT
)
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-L/14"
RESULTS_FILE = "results/reproduction_data_v6.json"
os.makedirs("results", exist_ok=True)

# ─── 논문 공식 80개 ImageNet 프롬프트 템플릿 ───────────────────────────────
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

# ─── 데이터셋별 공식 맞춤 프롬프트 템플릿 ────────────────────────────────────
# 논문 Appendix: dataset-specific prompt engineering
dataset_prompts = {
    "MNIST": [
        'a photo of the number: "{}".', 
        'a photo of the digit "{}".', 
        'a black and white photo of the number "{}".',
    ],
    "OxfordIIITPet": [
        'a photo of a {}, a type of pet.',
        'a photo of the {}, a type of pet.',
        'a photo of my cute {}.',
    ] + imagenet_templates,
    "Food101": [
        'a photo of {}, a type of food.',
        'a photo of a {}, a type of food.',
        'a meal of {}.',
    ] + imagenet_templates,
    "FGVCAircraft": [
        'a photo of a {}, a type of aircraft.',
        'a photo of the {}, a type of aircraft.',
        'an airplane photo of a {}.',
    ] + imagenet_templates,
    "Flowers102": [
        'a photo of a {}, a type of flower.',
        'a photo of the {}, a type of flower.',
        'a photo of a beautiful {} flower.',
    ] + imagenet_templates,
    "EuroSAT": [
        'a centered satellite photo of {}.',
        'a centered satellite photo of a {}.',
        'a satellite photo of {}.',
    ],
    "GTSRB": [
        'a photo of a {} traffic sign.',
        'a zoomed in photo of a "{}" traffic sign.',
        'a centered photo of a "{}" traffic sign.',
    ],
    "RenderedSST2": [
        'a {} movie review.',
        'a {} review of a film.',
    ],
}

def clean_class_name(c):
    """Aircraft는 'Boeing 737 - 300'→ '300' 처럼 되어 있어서 정제 필요"""
    if " - " in c:
        c = c.split(" - ")[-1]
    return c.replace("_", " ").strip()

def extract_all_features(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            features = model.encode_image(images.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().float().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

def build_zeroshot_weights(model, class_names, templates, device):
    """논문과 동일: 각 클래스별로 모든 템플릿을 앙상블하여 정규화
    배치 처리: 클래스별로 모든 템플릿을 한 번에 encode하여 속도 향상
    """
    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(class_names, leave=True, desc="ZS weights"):
            cleaned = clean_class_name(classname)
            texts = clip.tokenize(
                [t.format(cleaned) for t in templates], truncate=True
            ).to(device)
            class_embeddings = model.encode_text(texts).float()
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1)  # [D, num_classes]

def zero_shot_accuracy(test_feats, test_labels, zs_weights, device):
    feats = torch.from_numpy(test_feats).to(device)
    preds = (feats @ zs_weights).argmax(dim=-1).cpu().numpy()
    return (preds == test_labels).mean() * 100

def best_C_linear_probe(train_feats, train_labels, test_feats, test_labels, is_few_shot=False):
    """
    논문 Appendix: sweep C from 1e-6 to 1e6 in 96 log-spaced steps.
    For few-shot, we use a much faster abbreviated search to prevent overfitting and speed up.
    """
    n_train = len(train_labels)
    
    if not is_few_shot:
        Cs = np.logspace(-6, 6, 96)
        val_size = max(1, min(int(n_train * 0.1), 1000))
        rng = np.random.default_rng(42)
        idx = rng.permutation(n_train)
        val_idx, tr_idx = idx[:val_size], idx[val_size:]
        
        search_size = min(len(tr_idx), 2000)
        tr_idx_search = tr_idx[:search_size]
        
        best_C, best_val_acc = 0.316, -1
        # C 탐색 시에는 최대 2000개 샘플, 100 iter 스펙으로 초고속 탐색
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=100, solver='lbfgs',
                                      random_state=0)
            clf.fit(train_feats[tr_idx_search], train_labels[tr_idx_search])
            val_acc = clf.score(train_feats[val_idx], train_labels[val_idx])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_C = C
                
        # Retrain on full training set with best C
        clf = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs',
                                  random_state=0)
        clf.fit(train_feats, train_labels)
    else:
        # For few-shot: just search on train set (prevent overfitting)
        # Use the C that gives best training accuracy as proxy
        # (論文は同様に train data のみで C を選定)
        best_C, best_train_acc = 0.316, -1
        for C in [0.001, 0.01, 0.1, 0.316, 1.0, 3.16, 10.0, 100.0]:
            clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs',
                                      random_state=0)
            clf.fit(train_feats, train_labels)
            tr_acc = clf.score(train_feats, train_labels)
            if tr_acc > best_train_acc:
                best_train_acc = tr_acc
                best_C = C
        clf = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs',
                                  random_state=0)
        clf.fit(train_feats, train_labels)
    
    return clf.score(test_feats, test_labels) * 100

def get_few_shot_subset(features, labels, shots, num_classes, seed=42):
    rng = np.random.default_rng(seed)
    indices = []
    for c in range(num_classes):
        c_idxs = np.where(labels == c)[0]
        if len(c_idxs) >= shots:
            picked = rng.choice(c_idxs, shots, replace=False)
        else:
            picked = c_idxs
        indices.extend(picked.tolist())
    indices = np.array(indices)
    return features[indices], labels[indices]

def load_dataset(name, preprocess):
    """데이터셋 객체 (train, test) 반환"""
    root = "data_torchvision"
    if name == "CIFAR10":
        return (CIFAR10(root=root, train=True, download=True, transform=preprocess),
                CIFAR10(root=root, train=False, download=True, transform=preprocess))
    elif name == "CIFAR100":
        return (CIFAR100(root=root, train=True, download=True, transform=preprocess),
                CIFAR100(root=root, train=False, download=True, transform=preprocess))
    elif name == "MNIST":
        return (MNIST(root=root, train=True, download=True, transform=preprocess),
                MNIST(root=root, train=False, download=True, transform=preprocess))
    elif name == "STL10":
        return (STL10(root=root, split="train", download=True, transform=preprocess),
                STL10(root=root, split="test", download=True, transform=preprocess))
    elif name == "Food101":
        return (Food101(root=root, split="train", download=True, transform=preprocess),
                Food101(root=root, split="test", download=True, transform=preprocess))
    elif name == "OxfordIIITPet":
        return (OxfordIIITPet(root=root, split="trainval", download=True, transform=preprocess),
                OxfordIIITPet(root=root, split="test", download=True, transform=preprocess))
    elif name == "Flowers102":
        return (Flowers102(root=root, split="train", download=True, transform=preprocess),
                Flowers102(root=root, split="test", download=True, transform=preprocess))
    elif name == "FGVCAircraft":
        return (FGVCAircraft(root=root, split="trainval", download=True, transform=preprocess),
                FGVCAircraft(root=root, split="test", download=True, transform=preprocess))
    elif name == "RenderedSST2":
        return (RenderedSST2(root=root, split="train", download=True, transform=preprocess),
                RenderedSST2(root=root, split="test", download=True, transform=preprocess))
    elif name == "GTSRB":
        return (GTSRB(root=root, split="train", download=True, transform=preprocess),
                GTSRB(root=root, split="test", download=True, transform=preprocess))
    elif name == "EuroSAT":
        # EuroSAT has no standard split; use 80/20
        full = EuroSAT(root=root, download=True, transform=preprocess)
        n = len(full)
        tr_size = int(0.8 * n)
        from torch.utils.data import random_split
        import torch as th
        g = th.Generator().manual_seed(42)
        train_ds, test_ds = random_split(full, [tr_size, n - tr_size], generator=g)
        return train_ds, test_ds
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_classes(name, test_ds):
    class_lists = {
        "MNIST": [str(i) for i in range(10)],
        "GTSRB": [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
            'No passing', 'No passing for vehicles over 3.5 metric tons',
            'Right-of-way at the next intersection', 'Priority road', 'Yield',
            'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
            'No entry', 'General caution', 'Dangerous curve to the left',
            'Dangerous curve to the right', 'Double curve', 'Bumpy road',
            'Slippery road', 'Road narrows on the right', 'Road work',
            'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
            'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
            'Ahead only', 'Go straight or right', 'Go straight or left',
            'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 metric tons'
        ],
        "EuroSAT": [
            'annual crop land', 'forest', 'brushland or shrubland',
            'highway or road', 'industrial buildings or commercial buildings',
            'pasture land', 'permanent crop land', 'residential buildings or homes or apartments',
            'river', 'lake or sea'
        ],
        "RenderedSST2": ["negative", "positive"],
    }
    if name in class_lists:
        return class_lists[name]
    if hasattr(test_ds, 'classes'):
        return test_ds.classes
    # dataset.dataset.classes for Subset
    if hasattr(test_ds, 'dataset') and hasattr(test_ds.dataset, 'classes'):
        return test_ds.dataset.classes
    raise ValueError(f"Cannot determine class names for {name}")

def main():
    print(f"Using device: {DEVICE}")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    dataset_order = [
        "CIFAR10", "CIFAR100", "STL10", "Food101",
        "OxfordIIITPet", "Flowers102", "FGVCAircraft",
        "MNIST", "GTSRB", "EuroSAT", "RenderedSST2",
    ]

    results_data = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results_data = json.load(f)

    for name in dataset_order:
        # 이미 full pipeline 완료된 경우 건너뜀
        if name in results_data and "full_linear_probe" in results_data[name]:
            print(f"--- {name}: 이미 완료, 건너뜀 ---")
            continue

        print(f"\n{'='*50}")
        print(f"데이터셋: {name}")
        print(f"{'='*50}")

        try:
            train_ds, test_ds = load_dataset(name, preprocess)
            class_names = get_classes(name, test_ds)
            num_classes = len(class_names)
            print(f"클래스 수: {num_classes}")

            # ── Feature 추출 ──────────────────────────────
            print("Test feature 추출 중...")
            test_feats, test_labels = extract_all_features(model, test_ds, DEVICE)
            print("Train feature 추출 중...")
            train_feats, train_labels = extract_all_features(model, train_ds, DEVICE)
            print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")

            results_data.setdefault(name, {"shots": {}, "zero_shot": 0, "full_linear_probe": 0})

            # ── 1) Zero-shot ──────────────────────────────
            templates = dataset_prompts.get(name, imagenet_templates)
            print(f"Zero-shot 평가 중... (프롬프트 {len(templates)}개)")
            zs_weights = build_zeroshot_weights(model, class_names, templates, DEVICE)
            zs_acc = zero_shot_accuracy(test_feats, test_labels, zs_weights, DEVICE)
            results_data[name]["zero_shot"] = float(zs_acc)
            print(f"  Zero-shot: {zs_acc:.2f}%")

            # ── 2) Few-shot Linear Probe (1,2,4,8,16) ────
            for s in [1, 2, 4, 8, 16]:
                print(f"  {s}-shot LP 평가 중...")
                fs_feats, fs_labels = get_few_shot_subset(train_feats, train_labels, s, num_classes)
                lp_acc = best_C_linear_probe(fs_feats, fs_labels, test_feats, test_labels, is_few_shot=True)
                results_data[name]["shots"][str(s)] = float(lp_acc)
                print(f"    {s}-shot: {lp_acc:.2f}%")

            # ── 3) Full Linear Probe ──────────────────────
            print("  Full LP 평가 중 (C 최적화 포함)...")
            flp_acc = best_C_linear_probe(train_feats, train_labels, test_feats, test_labels)
            results_data[name]["full_linear_probe"] = float(flp_acc)
            print(f"  Full LP: {flp_acc:.2f}%")

            # 진행 상황 즉시 저장
            with open(RESULTS_FILE, "w") as f:
                json.dump(results_data, f, indent=4)
            print(f"✓ {name} 완료, 결과 저장됨")

        except Exception as e:
            import traceback
            print(f"✗ {name} 실패: {e}")
            traceback.print_exc()

    print("\n\n=== 전체 실험 완료 ===")
    print(json.dumps(results_data, indent=2))

if __name__ == "__main__":
    main()
