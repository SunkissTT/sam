"""
V5 시각화 스크립트 — 논문 Figure 6, 7, 8 재현
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

RESULTS_FILE = "results/reproduction_data_v6.json"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

# 유효한 데이터셋만 (zero_shot > 0 이고 shots 5개 모두 존재하고 full_lp 존재)
valid = {k: v for k, v in data.items()
         if v.get("zero_shot", 0) > 0
         and len(v.get("shots", {})) == 5
         and "full_linear_probe" in v}

print("Valid datasets:", list(valid.keys()))
if not valid:
    print("No valid data found.")
    exit(1)

SHOT_VALUES = [1, 2, 4, 8, 16]
LOG_SHOTS   = np.log2(SHOT_VALUES)

# ────────────────────────────────────────────────────────────
# Figure 6: Zero-shot vs Few-shot Linear Probe
# ────────────────────────────────────────────────────────────
def plot_figure_6():
    avg_zs = np.mean([v["zero_shot"] for v in valid.values()])
    avg_lp = [np.mean([v["shots"][str(s)] for v in valid.values()]) for s in SHOT_VALUES]

    fig, ax = plt.subplots(figsize=(8, 6))

    # X축 간격 실제 크기 반영 (Linear scale)
    # CLIP Linear Probe 커브
    ax.plot(SHOT_VALUES, avg_lp, 'o-', color='purple', label='Linear Probe CLIP (ViT-L/14)')

    # Zero-shot 점 (x=0 위치에 별표)
    ax.scatter([0], [avg_zs], marker='*', s=300, color='purple', zorder=5)
    ax.annotate(f'Zero-Shot\nCLIP\n({avg_zs:.1f}%)', xy=(0, avg_zs),
                xytext=(0.5, avg_zs - 4), fontsize=9, color='purple')

    # 점선 삭제 요청 반영 완료

    ax.set_xticks([0] + SHOT_VALUES)
    ax.set_xticklabels(['0', '1', '2', '4', '8', '16'])
    ax.set_xlabel('# of labeled training examples per class', fontsize=12)
    ax.set_ylabel('Average Score (%)', fontsize=12)
    ax.set_title('Figure 6: Zero-shot CLIP vs Few-shot Linear Probe', fontsize=13)
    ax.set_ylim(25, 100)
    ax.set_xlim(-1, 17)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'figure6_v6.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Figure 6 saved → {out}")

# ────────────────────────────────────────────────────────────
# Figure 7: Data Efficiency (log-linear interpolation)
# ────────────────────────────────────────────────────────────
def loglinear_interpolate(zs_acc, shot_accs, shot_vals, full_lp_acc, full_train_size, num_classes):
    """
    Log-linear interpolation over:
      x-axis : log2([1, 2, 4, 8, 16, full_per_class])
      y-axis : accuracy at each point
    Returns the estimated number of examples per class to reach zs_acc.
    """
    full_per_class = full_train_size / max(num_classes, 1)
    xs = np.array([0] + list(np.log2(shot_vals)) + [np.log2(full_per_class)])
    ys = np.array([shot_accs[0]] + list(shot_accs) + [full_lp_acc])

    # ys[0]이 1-shot이므로 xs[0] = 0(=log2(1)=0)... 이미 1=2^0
    xs = np.log2(np.array([1, 2, 4, 8, 16, full_per_class]))
    ys = np.array([shot_accs[0]] + list(shot_accs[1:]) + [full_lp_acc])

    # zero-shot이 모든 점보다 낮으면 < 1 shot
    if zs_acc <= ys[0]:
        return 0.5  # < 1 shot

    # zero-shot이 전체 full LP보다 크면 추정 불가
    if zs_acc >= ys[-1]:
        return full_per_class * 2  # 초과 표시용

    # 보간
    for i in range(len(ys) - 1):
        if ys[i] <= zs_acc <= ys[i + 1]:
            ratio = (zs_acc - ys[i]) / (ys[i + 1] - ys[i] + 1e-9)
            log_eff = xs[i] + ratio * (xs[i + 1] - xs[i])
            return 2 ** log_eff

    return full_per_class * 2

def get_train_size(name):
    """각 dataset의 train set 크기 (approximate)"""
    sizes = {
        "CIFAR10": 50000, "CIFAR100": 50000, "STL10": 5000,
        "Food101": 75750, "OxfordIIITPet": 3680, "Flowers102": 1020,
        "FGVCAircraft": 6667, "MNIST": 60000, "GTSRB": 39202,
        "EuroSAT": 21600, "RenderedSST2": 7792,
    }
    return sizes.get(name, 10000)

def get_num_classes(name):
    nc = {
        "CIFAR10": 10, "CIFAR100": 100, "STL10": 10,
        "Food101": 101, "OxfordIIITPet": 37, "Flowers102": 102,
        "FGVCAircraft": 100, "MNIST": 10, "GTSRB": 43,
        "EuroSAT": 10, "RenderedSST2": 2,
    }
    return nc.get(name, 10)

def plot_figure_7():
    excluded_fig7 = {"RenderedSST2"}
    results = []
    for name, v in valid.items():
        if name in excluded_fig7:
            continue
        zs = v["zero_shot"]
        sa = [v["shots"][str(s)] for s in SHOT_VALUES]
        flp = v["full_linear_probe"]
        tr_size = get_train_size(name)
        nc = get_num_classes(name)
        eff = loglinear_interpolate(zs, sa, SHOT_VALUES, flp, tr_size, nc)
        display = name.replace("OxfordIIITPet", "OxfordPets")
        results.append((display, eff))


    results.sort(key=lambda x: x[1])

    names = [r[0] for r in results]
    vals  = [r[1] for r in results]
    max_val = max(vals) * 1.15

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(names, vals, color='steelblue', edgecolor='black')

    for bar, val in zip(bars, vals):
        lbl = f'{val:.1f}'
        ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                lbl, va='center', fontsize=9)

    ax.set_xlim(0, max_val)
    ax.set_xlabel('# of labeled examples per class required to match zero-shot', fontsize=11)
    ax.set_title('Figure 7: Data Efficiency of Zero-shot Transfer\n(Log-Linear Interpolation incl. Full Supervision)',
                 fontsize=12)
    ax.grid(axis='x', ls='--', alpha=0.5)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, 'figure7_v6.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Figure 7 saved → {out}")

# ────────────────────────────────────────────────────────────
# Figure 8: Zero-shot vs Full LP Correlation
# ────────────────────────────────────────────────────────────
def plot_figure_8():
    excluded = {"RenderedSST2"}
    ds = {k: v for k, v in valid.items() if k not in excluded}

    if len(ds) < 2:
        print("Not enough data for Figure 8")
        return

    lp_accs = [v["full_linear_probe"] for v in ds.values()]
    zs_accs = [v["zero_shot"] for v in ds.values()]
    names   = [k.replace("OxfordIIITPet", "OxfordPets") for k in ds.keys()]

    r = np.corrcoef(lp_accs, zs_accs)[0, 1]

    # Regression line + CI
    xs = np.array(lp_accs)
    ys = np.array(zs_accs)
    m, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs) - 5, max(xs) + 5, 200)
    y_line = m * x_line + b

    # 95% CI (bootstrap)
    n_boot = 1000
    preds_boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), len(xs))
        mb = np.polyfit(xs[idx], ys[idx], 1)
        preds_boot.append(mb[0] * x_line + mb[1])
    preds_boot = np.array(preds_boot)
    ci_lo = np.percentile(preds_boot, 2.5, axis=0)
    ci_hi = np.percentile(preds_boot, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(lp_accs, zs_accs, s=60, color='steelblue', zorder=5, alpha=0.85)
    ax.plot(x_line, y_line, '-', color='steelblue', lw=2, label=f'Regression (r={r:.2f})')
    # 파란색 음영(CI) 흰색/투명으로 처리 요청 -> 사실상 삭제로 깔끔하게 처리
    # ax.fill_between(x_line, ci_lo, ci_hi, alpha=0.0, color='white')
    ax.plot([10, 100], [10, 100], 'k--', alpha=0.4, label='y = x')

    for i, name in enumerate(names):
        ax.annotate(name, (lp_accs[i], zs_accs[i]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlim(10, 105)
    ax.set_ylim(10, 105)
    ax.set_xlabel('Linear Probe CLIP Performance (%)', fontsize=12)
    ax.set_ylabel('Zero-Shot CLIP Performance (%)',   fontsize=12)
    ax.set_title(f'Figure 8: Zero-shot vs Linear Probe (r={r:.2f})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.35)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, 'figure8_v6.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Figure 8 saved → {out}")

if __name__ == "__main__":
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    print("All figures saved.")
