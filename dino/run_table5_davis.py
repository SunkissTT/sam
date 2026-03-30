#!/usr/bin/env python3
"""
DINO Table 5: DAVIS 2017 Video Object Segmentation 재현
eval_video_segmentation.py → J_m, F_m 계산 → 실시간 status.json 업데이트

Table 5 모델 (논문 기준):
  ViT-S/16: (J&F)_m=61.8, J_m=60.2, F_m=63.4
  ViT-B/16: (J&F)_m=62.3, J_m=60.7, F_m=63.9
  ViT-S/8:  (J&F)_m=69.9, J_m=66.6, F_m=73.1
  ViT-B/8:  (J&F)_m=71.4, J_m=67.9, F_m=74.9
"""
import subprocess, json, os, re, sys, time, glob
import numpy as np
from datetime import datetime
from pathlib import Path

# ─── 경로 설정 ───────────────────────────────────────────────
DINO   = "/home/daejun/shi_2026/dino"
RESULT = os.path.join(DINO, "results")
PT     = os.path.join(DINO, "pretrained")
PYTHON = "/home/daejun/miniconda3/envs/sam/bin/python"
DAVIS_PATH = os.path.join(DINO, "data/davis-2017/data/DAVIS")
STATUS_FILE = os.path.join(RESULT, "table5_davis_status.json")
EVAL_SCRIPT = os.path.join(DINO, "eval_video_segmentation.py")

# ─── Table 5 모델 목록 ───────────────────────────────────────
JOBS = [
    ("vits16", "vit_small", 16, f"{PT}/dino_vits16_imnet.pth"),
    ("vitb16", "vit_base",  16, f"{PT}/dino_vitb16_imnet.pth"),
    ("vits8",  "vit_small",  8, f"{PT}/dino_vits8_imnet.pth"),
    ("vitb8",  "vit_base",   8, f"{PT}/dino_vitb8_imnet.pth"),
]

# ─── 논문 Table 5 참조값 ─────────────────────────────────────
PAPER_VALUES = {
    "vits16": {"jf_m": 61.8, "j_m": 60.2, "f_m": 63.4},
    "vitb16": {"jf_m": 62.3, "j_m": 60.7, "f_m": 63.9},
    "vits8":  {"jf_m": 69.9, "j_m": 66.6, "f_m": 73.1},
    "vitb8":  {"jf_m": 71.4, "j_m": 67.9, "f_m": 74.9},
}

# ─── DAVIS val video 목록 ────────────────────────────────────
def get_val_videos():
    val_file = os.path.join(DAVIS_PATH, "ImageSets/2017/val.txt")
    with open(val_file) as f:
        return [l.strip() for l in f if l.strip()]


def write_status(state):
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATUS_FILE)


def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# J_m / F_m 계산 (DAVIS 공식 평가 방법)
# ═══════════════════════════════════════════════════════════════
def db_eval_iou(annotation, segmentation):
    """Jaccard index (J) = IoU between annotation and segmentation."""
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1.0
    return np.sum(annotation & segmentation) / np.sum(annotation | segmentation)


def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    """Contour-based accuracy (F). Simplified boundary F-measure."""
    from scipy.ndimage import binary_dilation, binary_erosion
    
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1.0
    if np.isclose(np.sum(annotation), 0) or np.isclose(np.sum(segmentation), 0):
        return 0.0

    # Compute boundary pixels
    bound_pix = max(1, round(bound_th * np.sqrt(annotation.shape[0] * annotation.shape[1])))
    
    # Get boundaries via erosion
    fg_boundary = annotation ^ binary_erosion(annotation, iterations=1)
    gt_boundary = annotation ^ binary_erosion(annotation, iterations=1)
    seg_boundary = segmentation ^ binary_erosion(segmentation, iterations=1)
    
    # Dilate boundaries
    fg_dil = binary_dilation(gt_boundary, iterations=bound_pix)
    seg_dil = binary_dilation(seg_boundary, iterations=bound_pix)
    
    # Precision and recall
    gt_match = np.sum(gt_boundary & seg_dil)
    seg_match = np.sum(seg_boundary & fg_dil)
    
    n_gt = np.sum(gt_boundary)
    n_seg = np.sum(seg_boundary)
    
    if n_gt == 0 and n_seg == 0:
        return 1.0
    if n_gt == 0 or n_seg == 0:
        return 0.0
        
    precision = gt_match / n_gt
    recall = seg_match / n_seg
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(pred_dir, gt_dir, val_videos):
    """모든 비디오에 대해 J_m, F_m 계산"""
    from PIL import Image
    
    j_scores = []
    f_scores = []
    
    for video in val_videos:
        pred_video_dir = os.path.join(pred_dir, video)
        gt_video_dir = os.path.join(gt_dir, video)
        
        if not os.path.isdir(pred_video_dir):
            print(f"  ⚠️ 예측 없음: {video}")
            continue
        
        gt_files = sorted(glob.glob(os.path.join(gt_video_dir, "*.png")))
        # Skip first frame (used as reference)
        for gt_file in gt_files[1:]:
            frame_name = os.path.basename(gt_file)
            pred_file = os.path.join(pred_video_dir, frame_name)
            
            if not os.path.exists(pred_file):
                continue
            
            gt_mask = np.array(Image.open(gt_file))
            pred_mask = np.array(Image.open(pred_file))
            
            # Handle multi-object: evaluate per-object and average
            obj_ids = np.unique(gt_mask)
            obj_ids = obj_ids[obj_ids != 0]  # remove background
            
            if len(obj_ids) == 0:
                continue
            
            frame_j = []
            frame_f = []
            for obj_id in obj_ids:
                gt_obj = (gt_mask == obj_id).astype(np.uint8)
                pred_obj = (pred_mask == obj_id).astype(np.uint8)
                frame_j.append(db_eval_iou(gt_obj, pred_obj))
                frame_f.append(db_eval_boundary(gt_obj, pred_obj))
            
            j_scores.append(np.mean(frame_j))
            f_scores.append(np.mean(frame_f))
    
    j_m = np.mean(j_scores) * 100 if j_scores else 0.0
    f_m = np.mean(f_scores) * 100 if f_scores else 0.0
    jf_m = (j_m + f_m) / 2
    
    return round(jf_m, 1), round(j_m, 1), round(f_m, 1)


# ═══════════════════════════════════════════════════════════════
# 단일 모델 평가 실행
# ═══════════════════════════════════════════════════════════════
def run_davis_eval(key, arch, patch, weights, state, val_videos):
    """단일 모델의 DAVIS 2017 세그멘테이션 평가"""
    output_dir = os.path.join(RESULT, f"davis_seg_{key}")
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    # CPU 모드 (RTX 5090 + PyTorch 1.7.1 비호환)
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        PYTHON, "-u", EVAL_SCRIPT,
        "--arch", arch,
        "--patch_size", str(patch),
        "--pretrained_weights", weights,
        "--data_path", DAVIS_PATH,
        "--output_dir", output_dir,
        "--n_last_frames", "7",
        "--size_mask_neighborhood", "12",
        "--topk", "5",
    ]

    log_path = os.path.join(RESULT, f"davis_eval_{key}.log")
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{ts}] START DAVIS: {key} ({arch} patch={patch})")
    sys.stdout.flush()

    total_videos = len(val_videos)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=DINO, env=env)
        accumulated = ""
        last_update = 0

        for line in proc.stdout:
            lf.write(line); lf.flush()
            accumulated += line
            print(line, end="", flush=True)

            # 비디오 진행상황 파싱: "[0/30] Begin to segmentate video bike-packing."
            m_vid = re.search(r"\[(\d+)/(\d+)\]\s+Begin to segmentate video\s+(\S+)", line)
            if m_vid:
                vid_idx = int(m_vid.group(1))
                vid_total = int(m_vid.group(2))
                vid_name = m_vid.group(3).rstrip(".")
                pct = round(vid_idx / max(vid_total, 1) * 95, 1)
                state[key]["phase"] = f"비디오 세그멘테이션: {vid_name}"
                state[key]["video_current"] = vid_idx
                state[key]["video_total"] = vid_total
                state[key]["progress_pct"] = pct
                now = time.time()
                if now - last_update >= 2:
                    write_status(state)
                    last_update = now

        proc.wait()

    return proc.returncode, accumulated, log_path, output_dir


def main():
    os.makedirs(RESULT, exist_ok=True)
    val_videos = get_val_videos()
    print(f"📋 DAVIS 2017 val: {len(val_videos)} 비디오")

    # 이전 상태 로드
    prev_state = load_status()
    state = {
        "table5_phase": "🔄 Table 5 DAVIS 실행중",
        "started_at": datetime.now().isoformat(),
        "log": "Table 5 DAVIS 2017 파이프라인 시작...\n",
        "paper_values": PAPER_VALUES,
        "total_videos": len(val_videos),
    }

    # 이전 완료 결과 복원
    skipped = []
    if prev_state:
        for key, *_ in JOBS:
            if key in prev_state and isinstance(prev_state[key], dict):
                if prev_state[key].get("status") == "done":
                    state[key] = prev_state[key]
                    skipped.append(key)
    if skipped:
        print(f"✅ 이전 결과 복원: {', '.join(skipped)}")
    write_status(state)

    for i, (key, arch, patch, weights) in enumerate(JOBS):
        if isinstance(state.get(key), dict) and state[key].get("status") == "done":
            r = state[key]
            print(f"⏭️ 건너뜀 (완료): {key} -> J&F={r.get('jf_m')}, J={r.get('j_m')}, F={r.get('f_m')}")
            continue

        state[key] = {"status": "running", "phase": "초기화", "progress_pct": 0}
        state["current"] = key
        state["current_idx"] = i
        state["total_jobs"] = len(JOBS)
        state["log"] = f"[{i+1}/{len(JOBS)}] DAVIS 평가: {key}\n"
        write_status(state)

        retcode, log_text, log_path, output_dir = run_davis_eval(key, arch, patch, weights, state, val_videos)

        if retcode == 0:
            # J_m, F_m 계산
            state[key]["phase"] = "J_m/F_m 계산중..."
            state[key]["progress_pct"] = 96
            write_status(state)

            gt_dir = os.path.join(DAVIS_PATH, "Annotations/480p")
            jf_m, j_m, f_m = evaluate_predictions(output_dir, gt_dir, val_videos)

            state[key] = {
                "status": "done",
                "jf_m": jf_m, "j_m": j_m, "f_m": f_m,
                "progress_pct": 100,
            }
            print(f"\n✅ {key}: (J&F)_m={jf_m}, J_m={j_m}, F_m={f_m}")
        else:
            state[key] = {"status": "error", "returncode": retcode, "progress_pct": 0}
            last_lines = "\n".join(log_text.splitlines()[-20:])
            state["log"] = last_lines + f"\n\n❌ Error (returncode={retcode})"
            print(f"\n❌ {key}: 오류 (retcode={retcode})")
        write_status(state)

    # ─── 완료 ──────────────────────────────────────────────
    state["table5_phase"] = "✅ Table 5 완료"
    summary = "\n[Table 5: DAVIS 2017 결과 요약]\n"
    summary += f"{'모델':<10} {'(J&F)_m':>8} {'J_m':>6} {'F_m':>6}  |  {'논문(J&F)':>9} {'논문J':>6} {'논문F':>6}\n"
    summary += "-" * 68 + "\n"
    for key, *_ in JOBS:
        s = state.get(key, {})
        p = PAPER_VALUES.get(key, {})
        if isinstance(s, dict) and s.get("status") == "done":
            summary += f"  {key:<10} {s['jf_m']:>7.1f}  {s['j_m']:>5.1f}  {s['f_m']:>5.1f}  |  {p.get('jf_m','?'):>8}  {p.get('j_m','?'):>5}  {p.get('f_m','?'):>5}\n"
        else:
            summary += f"  {key:<10} {'ERR':>7}  {'ERR':>5}  {'ERR':>5}  |  {p.get('jf_m','?'):>8}  {p.get('j_m','?'):>5}  {p.get('f_m','?'):>5}\n"

    state["log"] = "모든 Table 5 DAVIS 평가 완료!\n" + summary
    write_status(state)
    print("\n===== Table 5 DAVIS All Done =====")
    print(summary)


if __name__ == "__main__":
    main()
