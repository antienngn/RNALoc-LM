"""
Vẽ biểu đồ attention weights từ 2 CSV dump bởi base.py:239
- {rna}_test_textcnnbilstmattention_weights.csv: shape (113*5 or 97*5, 1024)
Output: figures/attention_*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/home/antn/RNALoc-LM"
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

# (rna_type, csv_path, n_test, actual_seq_len_max_in_data, zoom_pos)
CASES = [
    ("miRNA", "miRNA_test_textcnnbilstmattention_weights.csv", 113, 144, 200),
    ("lncRNA", "lncRNA_test_textcnnbilstmattention_weights.csv", 97, 1022, 1024),
]


def load_attn(path):
    df = pd.read_csv(path)
    mask = pd.to_numeric(df["Unnamed: 0"], errors="coerce").notna()
    df = df[mask].drop(columns=["Unnamed: 0"]).astype(float).reset_index(drop=True)
    return df.values


def fig_per_rna(rna, attn_path, n_test, seq_len_max, zoom_pos):
    arr = load_attn(os.path.join(REPO, attn_path))      # (n_test*5, L)
    n_folds = 5
    assert arr.shape[0] == n_test * n_folds, (rna, arr.shape, n_test, n_folds)
    folds = arr.reshape(n_folds, n_test, -1)            # (5, n_test, L)
    L = folds.shape[-1]
    uniform = 1.0 / L

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.4])
    fig.suptitle(f"Attention weights — {rna}  (5 fold × {n_test} test seq, L={L})",
                 fontsize=14, fontweight="bold")

    # Panel 1: mean attention curve per fold (full range)
    ax1 = fig.add_subplot(gs[0, :])
    for f in range(n_folds):
        ax1.plot(folds[f].mean(axis=0), alpha=0.6, lw=1, label=f"fold {f+1}")
    global_mean = arr.mean(axis=0)
    ax1.plot(global_mean, color="black", lw=2, label="mean (all folds)")
    ax1.axhline(uniform, color="red", ls="--", lw=1, label=f"uniform = 1/{L}")
    if seq_len_max < L:
        ax1.axvline(seq_len_max, color="gray", ls=":", lw=1.5,
                    label=f"max seq length in data ({seq_len_max})")
    ax1.set_xlim(0, L - 1)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Mean attention weight")
    ax1.set_title("Mean attention per position (per fold + overall)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # Panel 2: zoom vào vùng có data thật
    ax2 = fig.add_subplot(gs[1, 0])
    for f in range(n_folds):
        ax2.plot(folds[f].mean(axis=0)[:zoom_pos], alpha=0.6, lw=1, label=f"fold {f+1}")
    ax2.plot(global_mean[:zoom_pos], color="black", lw=2, label="mean")
    ax2.axhline(uniform, color="red", ls="--", lw=1)
    if seq_len_max < zoom_pos:
        ax2.axvline(seq_len_max, color="gray", ls=":", lw=1.5)
    ax2.set_xlabel("Position (zoom)")
    ax2.set_ylabel("Mean attention")
    ax2.set_title(f"Zoom pos 0–{zoom_pos} (vùng data thật)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: phân phối attention mass theo bin
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.array_split(np.arange(L), 8)
    masses = [global_mean[b].sum() for b in bins]
    labels = [f"{b[0]}–{b[-1]}" for b in bins]
    bars = ax3.bar(labels, masses, color="steelblue", edgecolor="black")
    for bar, m in zip(bars, masses):
        ax3.text(bar.get_x() + bar.get_width()/2, m, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax3.axhline(1/len(bins), color="red", ls="--", lw=1,
                label=f"uniform mass = 1/{len(bins)}")
    ax3.set_ylabel("Total attention mass")
    ax3.set_xlabel("Position bin")
    ax3.set_title("Attention mass theo vùng (sum over each bin)")
    ax3.tick_params(axis="x", rotation=30)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")

    # Panel 4: heatmap (samples × positions) — average across folds
    ax4 = fig.add_subplot(gs[2, :])
    avg_across_folds = folds.mean(axis=0)                # (n_test, L)
    # Subsample positions cho dễ nhìn nếu L lớn
    step = max(1, L // 256)
    img = avg_across_folds[:, ::step]
    sns.heatmap(img, ax=ax4, cmap="viridis", cbar_kws={"label": "attention"})
    ax4.set_xlabel(f"Position (mỗi tick = {step} positions)")
    ax4.set_ylabel("Test sample idx")
    ax4.set_title(f"Heatmap attention per sample (avg across {n_folds} folds)")

    out = os.path.join(OUT, f"attention_{rna}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return arr


def fig_compare(arrs):
    """So sánh miRNA vs lncRNA trong 1 figure"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("So sánh attention: miRNA vs lncRNA", fontsize=14, fontweight="bold")

    for ax, (rna, arr) in zip(axes, arrs.items()):
        L = arr.shape[1]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.fill_between(np.arange(L), mean - std, mean + std, alpha=0.25,
                        label="±1 std")
        ax.plot(mean, color="black", lw=1.5, label="mean")
        ax.axhline(1/L, color="red", ls="--", lw=1, label=f"uniform = 1/{L}")
        ax.set_title(rna)
        ax.set_xlabel("Position")
        ax.set_ylabel("Attention weight")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    out = os.path.join(OUT, "attention_compare.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    sns.set_theme(style="whitegrid", context="paper")
    print("Generating figures into", OUT)
    arrs = {}
    for rna, csv, n_test, seq_len_max, zoom in CASES:
        print(f"[{rna}]")
        arrs[rna] = fig_per_rna(rna, csv, n_test, seq_len_max, zoom)
    print("[compare]")
    fig_compare(arrs)
    print("Done.")


if __name__ == "__main__":
    main()
