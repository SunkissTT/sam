#!/usr/bin/env python3
"""
DINO Table 3 실행 + 실시간 status.json 업데이트 (재개 기능 포함)
이미 완료된 job은 status.json에서 읽어서 건너뜀.
CPU 모드 사용 (RTX 5090 sm_120 미지원 우회)
"""
import subprocess, json, os, re, sys, time
from datetime import datetime

RESULT = "/home/daejun/shi_2026/dino/results"
DINO   = "/home/daejun/shi_2026/dino"
PYTHON = "/home/daejun/miniconda3/envs/sam/bin/python"
PT     = os.path.join(DINO, "pretrained")
DATA   = os.path.join(DINO, "data/retrieval/datasets")
STATUS_FILE = os.path.join(RESULT, "status.json")

JOBS = [
    ("vits16imnet-oxford", "vit_small", 16, f"{PT}/dino_vits16_imnet.pth",    224, False, "roxford5k"),
    ("vits16imnet-paris",  "vit_small", 16, f"{PT}/dino_vits16_imnet.pth",    512, True,  "rparis6k"),
    ("vits16gldv2-oxford", "vit_small", 16, f"{PT}/dino_vits16_gldv2.pth",   224, False, "roxford5k"),
    ("vits16gldv2-paris",  "vit_small", 16, f"{PT}/dino_vits16_gldv2.pth",   512, True,  "rparis6k"),
    ("resnet50-oxford",    "resnet50",  16, f"{PT}/dino_resnet50_imnet.pth",  224, False, "roxford5k"),
    ("resnet50-paris",     "resnet50",  16, f"{PT}/dino_resnet50_imnet.pth",  512, True,  "rparis6k"),
]

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

def parse_map(log_text):
    m = re.search(r"mAP M: ([\d.]+), H: ([\d.]+)", log_text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

def tail_lines(text, n=40):
    return "\n".join(text.splitlines()[-n:])

def run_eval(key, arch, patch, weights, imsize, multiscale, dataset, state):
    env = os.environ.copy()
    for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        PYTHON, "-u",
        os.path.join(DINO, "eval_image_retrieval.py"),
        "--arch", arch,
        "--patch_size", str(patch),
        "--pretrained_weights", weights,
        "--imsize", str(imsize),
        "--multiscale", "1" if multiscale else "0",
        "--data_path", DATA,
        "--dataset", dataset,
        "--use_cuda", "False",
    ]
    os.makedirs(RESULT, exist_ok=True)
    log_path = os.path.join(RESULT, f"eval_{key}.log")
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{ts}] START: {key} (CPU mode)")
    sys.stdout.flush()

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=DINO, env=env)
        accumulated = ""
        total_imgs = 0
        query_imgs = 0
        phase = "loading"
        last_update = 0

        for line in proc.stdout:
            lf.write(line); lf.flush()
            accumulated += line
            print(line, end="", flush=True)

            m_imgs = re.search(r"train: (\d+) imgs / query: (\d+) imgs", line)
            if m_imgs:
                total_imgs = int(m_imgs.group(1))
                query_imgs = int(m_imgs.group(2))
                state[key]["total_train"] = total_imgs
                state[key]["total_query"] = query_imgs
                state[key]["phase"] = "모델 로딩"
                write_status(state)

            if "Pretrained weights found" in line or "loaded with msg" in line:
                state[key]["phase"] = "train 특징 추출"
                write_status(state)

            m_prog = re.search(r"\[\s*(\d+)/(\d+)\]\s+eta:\s+(\S+)", line)
            if m_prog:
                current = int(m_prog.group(1))
                total = int(m_prog.group(2))
                eta = m_prog.group(3)

                if total == total_imgs or (total > 100 and phase != "query_extract"):
                    phase = "train_extract"
                    state[key]["phase"] = "train 특징 추출"
                    state[key]["train_current"] = current
                    state[key]["train_total"] = total
                elif total == query_imgs or total < 100:
                    phase = "query_extract"
                    state[key]["phase"] = "query 특징 추출"
                    state[key]["query_current"] = current
                    state[key]["query_total"] = total

                state[key]["eta"] = eta

                if phase == "train_extract" and total > 0:
                    pct = round(current / total * 80, 1)
                elif phase == "query_extract":
                    pct = 80 + round(current / max(total, 1) * 20, 1)
                else:
                    pct = 0
                state[key]["progress_pct"] = pct

                now = time.time()
                if now - last_update >= 2:
                    write_status(state)
                    last_update = now

            m_map = re.search(r"mAP M: ([\d.]+), H: ([\d.]+)", line)
            if m_map:
                state[key]["phase"] = "완료"
                state[key]["progress_pct"] = 100
                write_status(state)

        proc.wait()
    write_status(state)
    return proc.returncode, accumulated, log_path

def main():
    os.makedirs(RESULT, exist_ok=True)

    # 이전 상태 로드 — 이미 완료된 job 건너뛰기
    prev_state = load_status()
    state = {
        "table3_phase": "🔄 실행중 (CPU mode)",
        "started_at": datetime.now().isoformat(),
        "log": "Table 3 평가 재개...\n"
    }

    # 이전 결과 복원
    skipped = []
    for key, *_ in JOBS:
        if prev_state and key in prev_state and isinstance(prev_state[key], dict):
            if prev_state[key].get("status") == "done":
                state[key] = prev_state[key]  # 이전 결과 유지
                skipped.append(key)
                continue
        state[key] = {"status": "waiting"}

    if skipped:
        print(f"✅ 이전 결과 복원: {', '.join(skipped)}")
        state["log"] = f"이전 완료 결과 {len(skipped)}건 복원. 나머지 실행 시작.\n"
    write_status(state)

    for i, (key, arch, patch, weights, imsize, multiscale, dataset) in enumerate(JOBS):
        # 이미 완료된 job 건너뛰기
        if isinstance(state.get(key), dict) and state[key].get("status") == "done":
            print(f"⏭️ 건너뜀 (이미 완료): {key} -> mAP M={state[key]['mapM']}, H={state[key]['mapH']}")
            continue

        state[key] = {"status": "running", "phase": "초기화", "progress_pct": 0}
        state["current"] = key
        state["current_idx"] = i
        state["total_jobs"] = len(JOBS)
        state["log"] = f"[{i+1}/6] 실행중: {key}\n"
        write_status(state)

        retcode, log_text, log_path = run_eval(key, arch, patch, weights, imsize, multiscale, dataset, state)
        mapM, mapH = parse_map(log_text)

        if retcode == 0 and mapM is not None:
            state[key] = {"status": "done", "mapM": round(mapM, 2), "mapH": round(mapH, 2), "progress_pct": 100}
            state["log"] = tail_lines(log_text)
            print(f"\n✅ {key}: mAP M={mapM:.2f} H={mapH:.2f}")
        else:
            state[key] = {"status": "error", "returncode": retcode, "progress_pct": 0}
            state["log"] = tail_lines(log_text) + f"\n\n❌ Error (returncode={retcode})"
            print(f"\n❌ {key}: 오류 발생 (retcode={retcode})")
        write_status(state)

    state["table3_phase"] = "✅ Table 3 완료"
    summary = "\n[결과 요약]\n"
    for key, *_ in JOBS:
        s = state.get(key, {})
        if isinstance(s, dict) and s.get("status") == "done":
            summary += f"  {key}: mAP M={s['mapM']}, H={s['mapH']}\n"
        else:
            summary += f"  {key}: {s.get('status','?') if isinstance(s,dict) else s}\n"
    state["log"] = "모든 Table 3 평가 완료!\n" + summary
    write_status(state)
    print("\n===== Table 3 All Done =====")
    print(summary)

if __name__ == "__main__":
    main()
