# OpenCLIP Zero-shot & Few-shot Evaluation

이 저장소는 OpenCLIP 모델을 사용하여 CIFAR-100, Food-101 데이터셋에 대한 Zero-shot 및 Few-shot 성능 검증 실험을 수행한 환경을 포함하고 있습니다.

## 📋 실험 개요
- **모델**: OpenCLIP (ViT-B/32, laion2b_s34b_b79k)
- **데이터셋**: CIFAR-100, Food-101
- **주요 기능**:
    - Zero-shot 정확도 측정
    - Few-shot (16-shot) 성능 검증

## 🛠 설치 방법 (Setup)
1. 저장소 클론:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. 가상 환경 생성 및 활성화:
   ```bash
   python3 -m venv .env
   source .env/bin/activate
   ```
3. 필수 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 실행 방법 (Execution)
### 1. Zero-shot 성능 측정
```bash
python3 measure_accuracy.py          # CIFAR-100
python3 measure_accuracy_food101.py  # Food-101
```

### 2. Few-shot 성능 측정
```bash
python3 few_shot_cifar100.py
python3 few_shot_food101.py
```

## 📁 디렉토리 구조
- `measure_accuracy*.py`: 데이터셋별 Zero-shot 측정 스크립트
- `few_shot_*.py`: Few-shot 학습 및 평가 스크립트
- `requirements.txt`: 필요한 라이브러리 목록

## ⚠️ 주의 사항
- 대용량 데이터셋 및 가주치 파일(`.pth`, `data/`)은 저장소에 포함되어 있지 않으므로 별도 준비가 필요합니다.
