# DINO 논문 재현 실험 (Table 3 & Table 5)

---

## 실험 환경

| 항목 | 내용 |
|------|------|
| **OS** | Linux |
| **GPU** | NVIDIA RTX 5090 32GB |
| **실행 모드** | **CPU** (PyTorch 1.7.1과 RTX 5090 비호환으로 CPU 모드 사용) |
| **Python** | 3.10 (conda `sam` 환경) |
| **PyTorch** | 2.6.0 |

---

## 설치

```bash
pip install -r requirements.txt
```

---

## Pretrained Weights 다운로드

실험에 사용된 공식 DINO pretrained weights:

| 모델 | 다운로드 |
|------|----------|
| ViT-S/16 (ImageNet) | [링크](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth) |
| ViT-S/16 (GLDv2) | [링크](https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth) |
| ViT-S/8 (ImageNet) | [링크](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth) |
| ViT-B/16 (ImageNet) | [링크](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) |
| ViT-B/8 (ImageNet) | [링크](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth) |
| ResNet-50 (ImageNet) | [링크](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) |

다운로드한 파일을 `pretrained/` 폴더에 저장합니다.

---

## Table 3: Image Retrieval (이미지 검색)

### 데이터셋 준비

[revisitop](https://github.com/filipradenovic/revisitop) 저장소를 참고하여 ROxford5k와 RParis6k 데이터셋을 준비합니다.

```
data/retrieval/datasets/
  ├── roxford5k/
  └── rparis6k/
```

### 실험 실행

```bash
# 전체 자동 실행 (3개 모델 × 2개 데이터셋)
bash run_table3.sh

# 또는 상태 추적 포함 실행
python run_table3_with_status.py
```

개별 실행 예시:
```bash
# RParis6k (imsize=512, multiscale)
python eval_image_retrieval.py \
    --arch vit_small --patch_size 16 \
    --pretrained_weights pretrained/dino_vits16_imnet.pth \
    --imsize 512 --multiscale 1 \
    --data_path data/retrieval/datasets \
    --dataset rparis6k

# ROxford5k (imsize=224, no multiscale)
python eval_image_retrieval.py \
    --arch vit_small --patch_size 16 \
    --pretrained_weights pretrained/dino_vits16_imnet.pth \
    --imsize 224 --multiscale 0 \
    --data_path data/retrieval/datasets \
    --dataset roxford5k

---

## Table 5: DAVIS 2017 Video Object Segmentation

### 데이터셋 준비

```bash
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
```

데이터를 `data/davis-2017/` 경로에 위치시킵니다.

### 실험 실행

```bash
# 4개 모델 전체 자동 실행 (J_m, F_m 자동 계산 포함)
python run_table5_davis.py
```

개별 실행 예시:
```bash
python eval_video_segmentation.py \
    --arch vit_small --patch_size 16 \
    --pretrained_weights pretrained/dino_vits16_imnet.pth \
    --data_path data/davis-2017/data/DAVIS \
    --output_dir results/davis_seg_vits16 \
    --n_last_frames 7 \
    --size_mask_neighborhood 12 \
    --topk 5
```

## 파일 구조

```
dino/
├── eval_image_retrieval.py   # Table 3 평가 스크립트
├── eval_video_segmentation.py # Table 5 평가 스크립트
├── run_table3.sh             # Table 3 전체 실행 셸 스크립트
├── run_table3_with_status.py # Table 3 상태 추적 포함 실행
├── run_table5_davis.py       # Table 5 전체 실행 + J/F 계산
├── vision_transformer.py     # ViT 모델 구조
├── utils.py                  # 유틸리티 함수
├── requirements.txt          # 패키지 목록
├── pretrained/               # 모델 가중치 (별도 다운로드)
└── data/                     # 데이터셋 (별도 준비)
    ├── retrieval/datasets/
    │   ├── roxford5k/
    │   └── rparis6k/
    └── davis-2017/
```
