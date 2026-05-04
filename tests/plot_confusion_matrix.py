"""
Generates the combined confusion matrix for all 12 evaluated persons
(video1 + video2 + 5 CAVIAR videos).

Run:
    python tests/plot_confusion_matrix.py

Output:
    data/combined_confusion_matrix.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import classification_report

# ---------------------------------------------------------------------------
# Ground truth and predictions — 12 persons across all videos
# ---------------------------------------------------------------------------
# (ground_truth, prediction, description)
RESULTS = [
    ("HIGH",   "HIGH",   "video1 — P1"),
    ("HIGH",   "HIGH",   "video2 — P1"),
    ("HIGH",   "HIGH",   "WalkByShop — P1"),
    ("LOW",    "LOW",    "WalkByShop — P2"),
    ("LOW",    "LOW",    "WalkByShop — P3"),
    ("LOW",    "LOW",    "OneShopOneWait — P1"),
    ("LOW",    "LOW",    "OneShopOneWait — P2"),
    ("MEDIUM", "MEDIUM", "OneShopOneWait — P3"),
    ("LOW",    "LOW",    "OneLeaveShopReenter — P1"),
    ("HIGH",   "HIGH",    "OneLeaveShopReenter — P2"),
    ("LOW",    "LOW",    "OneLeaveShopReenter — P3"),
    ("MEDIUM", "HIGH",   "OneStopMoveEnter — P1"),   # system over-fired
    ("LOW",    "LOW",    "OneStopMoveEnter — P2"),
    ("NONE",   "NONE",   "OneStopMoveEnter — P3"),
    ("NONE",   "LOW",    "ShopAssistant — P1"),       # system over-fired
    ("MEDIUM", "MEDIUM", "ShopAssistant — P2"),
    ("HIGH",   "HIGH",   "ShopAssistant — P3"),
]

y_true = [r[0] for r in RESULTS]
y_pred = [r[1] for r in RESULTS]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
report   = classification_report(y_true, y_pred)
macro    = report.get("macro avg", {})

print(f"\n== Evaluation Results (n={len(y_true)}) ================================")
print(f"  Accuracy        : {accuracy:.1%}")
print(f"  Macro F1        : {macro.get('f1', 0):.3f}")
print(f"  Macro Precision : {macro.get('precision', 0):.3f}")
print(f"  Macro Recall    : {macro.get('recall', 0):.3f}")

print(f"\n== Per-class Report ================================================")
print(f"  {'Class':<10} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Support':>8}")
for cls, v in report.items():
    print(f"  {cls:<10} {v['precision']:>9.3f} {v['recall']:>7.3f}"
          f" {v['f1']:>7.3f} {v['support']:>8}")

print(f"\n== Per-person Results ==============================================")
print(f"  {'Description':<30} {'GT':<8} {'Pred':<8} {'OK'}")
print(f"  {'-'*58}")
for gt, pred, desc in RESULTS:
    ok = "PASS" if gt == pred else "FAIL"
    print(f"  {desc:<30} {gt:<8} {pred:<8} {ok}")

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
LEVELS  = ["NONE", "LOW", "MEDIUM", "HIGH"]
classes = [l for l in LEVELS if l in set(y_true) | set(y_pred)]
n       = len(classes)
idx     = {c: i for i, c in enumerate(classes)}
cm      = np.zeros((n, n), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[idx[t]][idx[p]] += 1

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

cmap    = plt.cm.Blues
max_val = max(cm.max(), 1)

for i in range(n):
    for j in range(n):
        val  = cm[i, j]
        norm = 0.15 + 0.85 * (val / max_val) if val > 0 else 0.15
        color = cmap(norm)
        rect  = patches.FancyBboxPatch(
            (j - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.02", linewidth=0, facecolor=color,
        )
        ax.add_patch(rect)
        lum       = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        txt_color = "white" if lum < 0.55 else "#0d1117"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=16, fontweight="bold", color=txt_color)

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(-0.5, n - 0.5)
ax.invert_yaxis()
ax.set_xticks(range(n));  ax.set_xticklabels(classes, color="white", fontsize=11)
ax.set_yticks(range(n));  ax.set_yticklabels(classes, color="white", fontsize=11)
ax.set_xlabel("Predicted",  color="#8b949e", fontsize=11)
ax.set_ylabel("True Label", color="#8b949e", fontsize=11)
ax.set_title("Confusion Matrix — All Ground Truths (n=12)",
             color="white", fontsize=12, pad=12)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_val))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
cb.outline.set_edgecolor("#30363d")

plt.tight_layout()
out = ROOT / "data" / "combined_confusion_matrix.png"
plt.savefig(out, dpi=150, facecolor="#0d1117")
plt.close(fig)
print(f"\n  Confusion matrix saved -> {out}\n")
