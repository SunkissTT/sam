# CLIP 논문 재현 실험 (Figure 6, 7, 8)

## 실험 환경

| 항목 | 내용 |
|------|------|
| **OS** | Linux |
| **GPU** | NVIDIA RTX 5090 32GB |
| **Python** | 3.10 |
| **PyTorch** | 2.x |
| **모델** | CLIP `ViT-L/14` (openai 공식 가중치) |

---

## 설치

```bash
pip install -r requirements.txt
```

> CLIP은 공식 pip 패키지가 없으므로 GitHub에서 직접 설치합니다.

---

## 실험 실행

### 1단계: 데이터 수집 및 평가 (`run_reproduction_v6.py`)

11개 데이터셋에 대해 Zero-shot, Few-shot (1/2/4/8/16-shot), Full Linear Probe를 자동으로 평가합니다.

```bash
python run_reproduction_v6.py
```

결과는 `results/reproduction_data_v6.json`에 저장됩니다.

**평가 대상 데이터셋** (torchvision 자동 다운로드):
- CIFAR-10, CIFAR-100, STL-10, MNIST
- Food101, OxfordIIITPet, Flowers102, FGVCAircraft
- GTSRB, EuroSAT, RenderedSST2

**핵심 설계**:
- **Zero-shot**: 데이터셋별 맞춤 프롬프트 + ImageNet 80개 템플릿 앙상블
- **Few-shot Linear Probe**: `LogisticRegression` (L2 정규화, C 자동 최적화)
- **Full Linear Probe**: C sweep (1e-6 ~ 1e6, 96 log-spaced)

### 2단계: 결과 시각화 (`visualize_results_v6.py`)

```bash
python visualize_results_v6.py
```

`figures/` 폴더에 Figure 6, 7, 8 이미지가 저장됩니다.

---

## 파일 구조

```
shi_2026/
├── run_reproduction_v6.py
├── visualize_results_v6.py
├── requirements.txt
├── figures/
├── results/
└── data_torchvision/      ← 최초 실행 시 자동 생성 (총 ~수십 GB)
    ├── cifar-10-batches-py/
    ├── food-101/
    └── ...

```
