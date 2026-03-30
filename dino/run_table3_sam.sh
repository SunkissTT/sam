#!/bin/bash
# DINO Table 3: Image Retrieval 전체 평가 스크립트 (sam env - PyTorch 2.6 + RTX 5090)

set -e

DATASETS_DIR=/home/daejun/shi_2026/dino/data/retrieval/datasets
PT=/home/daejun/shi_2026/dino/pretrained
RESULT=/home/daejun/shi_2026/dino/results
cd /home/daejun/shi_2026/dino

export MASTER_PORT=12360
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost

CONDA_PYTHON=/home/daejun/miniconda3/envs/sam/bin/python

echo "===== Table 3 Image Retrieval Evaluation (sam env) =====" | tee $RESULT/table3_results.txt

for MODEL in vits16_imnet vits16_gldv2 resnet50_imnet; do
    if [ "$MODEL" = "vits16_imnet" ]; then
        ARCH=vit_small; PATCH=16; WEIGHTS=$PT/dino_vits16_imnet.pth
    elif [ "$MODEL" = "vits16_gldv2" ]; then
        ARCH=vit_small; PATCH=16; WEIGHTS=$PT/dino_vits16_gldv2.pth
    elif [ "$MODEL" = "resnet50_imnet" ]; then
        ARCH=resnet50; PATCH=0; WEIGHTS=$PT/dino_resnet50_imnet.pth
    fi

    # ROxford5k (imsize=224, no multiscale)
    echo "" | tee -a $RESULT/table3_results.txt
    echo "=== $MODEL + ROxford5k ===" | tee -a $RESULT/table3_results.txt
    $CONDA_PYTHON eval_image_retrieval.py \
        --arch $ARCH --patch_size $PATCH \
        --pretrained_weights $WEIGHTS \
        --imsize 224 --multiscale 0 \
        --data_path $DATASETS_DIR \
        --dataset roxford5k 2>&1 | tee $RESULT/eval_${MODEL}_oxford.log | grep -E "mAP|>> Dataset|train:|Pretrained"
    grep "mAP" $RESULT/eval_${MODEL}_oxford.log >> $RESULT/table3_results.txt || true

    # RParis6k (imsize=512, multiscale)
    echo "" | tee -a $RESULT/table3_results.txt
    echo "=== $MODEL + RParis6k ===" | tee -a $RESULT/table3_results.txt
    $CONDA_PYTHON eval_image_retrieval.py \
        --arch $ARCH --patch_size $PATCH \
        --pretrained_weights $WEIGHTS \
        --imsize 512 --multiscale 1 \
        --data_path $DATASETS_DIR \
        --dataset rparis6k 2>&1 | tee $RESULT/eval_${MODEL}_paris.log | grep -E "mAP|>> Dataset|train:|Pretrained"
    grep "mAP" $RESULT/eval_${MODEL}_paris.log >> $RESULT/table3_results.txt || true
done

echo "" | tee -a $RESULT/table3_results.txt
echo "===== Done =====" | tee -a $RESULT/table3_results.txt
cat $RESULT/table3_results.txt
