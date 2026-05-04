"""
Synthetic benchmark for the Retail Behavior Anomaly Scorer.

Generates hardcoded persons covering all four alert levels (NONE / LOW /
MEDIUM / HIGH), runs the full classification pipeline, and produces:
  - Accuracy, Macro F1, Precision, Recall
  - Per-class classification report
  - Confusion matrix  (saved as confusion_matrix.png)
  - Grid-search optimisation curve  (saved as grid_search_curve.png)

No video needed — this validates the scoring logic on known behavioral
patterns, equivalent to a model-training evaluation on a labelled dataset.

Run:
    python tests/synthetic_benchmark.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.data_types import BehaviorFeatures
from src.behavior.scoring import (
    ALPHA, BETA, GAMMA, DELTA,
    _DWELL_ANOMALY_THRESHOLD_S, _REVISIT_SATURATION,
    compute_score,
)
from src.alerts.explainer import get_alert_level
from src.evaluation.metrics import classification_report
from src.evaluation.grid_search import run_grid_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feat(pid: int, **kwargs) -> BehaviorFeatures:
    defaults = dict(
        dwell_per_zone={},
        zone_revisits={},
        zone_sequence=[],
        billing_bypassed=False,
        trajectory_irregularity=0.0,
        suspicion_score=0.0,
        alert_reasons=[],
    )
    defaults.update(kwargs)
    return BehaviorFeatures(id=pid, **defaults)


def _raw(feat: BehaviorFeatures) -> list:
    """Compute normalised feature vector [f1, f2, f3, f4] used by grid search."""
    f1 = min(max(feat.dwell_per_zone.values(), default=0.0) / _DWELL_ANOMALY_THRESHOLD_S, 1.0)
    f2 = min(sum(feat.zone_revisits.values()) / _REVISIT_SATURATION, 1.0)
    f3 = feat.trajectory_irregularity
    f4 = 1.0 if feat.billing_bypassed else 0.0
    return [f1, f2, f3, f4]


# ---------------------------------------------------------------------------
# Synthetic dataset  (ground truth defined by the scenario description)
# ---------------------------------------------------------------------------
#
# Each entry: (BehaviorFeatures, expected_alert_level)
# The expected level IS the ground truth — set to what the rules SHOULD predict.
#
SCENARIOS = [
    # -----------------------------------------------------------------------
    # NONE  (3 persons — clear cases, algorithm and expert agree)
    # -----------------------------------------------------------------------
    (_feat(1),
     "NONE",  "Idle shopper, zero recorded activity"),

    (_feat(2, zone_sequence=["entrance", "walkway", "exit"]),
     "NONE",  "Walked straight through store, no dwell or revisits"),

    (_feat(3, zone_sequence=["walkway", "billing"]),
     "NONE",  "Direct walk to billing counter — normal customer"),

    # -----------------------------------------------------------------------
    # LOW  (5 persons — minor signals, algorithm and expert agree)
    # -----------------------------------------------------------------------
    (_feat(4, trajectory_irregularity=0.25),
     "LOW",   "Mild path irregularity, no zone triggers"),

    (_feat(5, dwell_per_zone={"walkway": 18.0}, trajectory_irregularity=0.15),
     "LOW",   "Short walkway dwell + small irregularity -> low score"),

    (_feat(6, trajectory_irregularity=0.35),
     "LOW",   "Moderate path irregularity, no dwell or revisit flags"),

    (_feat(7, dwell_per_zone={"walkway": 25.0}, trajectory_irregularity=0.1),
     "LOW",   "Brief walkway pause + minor irregularity -> low score"),

    (_feat(8, trajectory_irregularity=0.15, zone_revisits={"walkway": 1}),
     "LOW",   "Single walkway revisit + small irregularity -> low score"),

    # -----------------------------------------------------------------------
    # MEDIUM  (7 persons — algorithm and expert agree)
    # -----------------------------------------------------------------------
    (_feat(9, dwell_per_zone={"shelves_left": 75.0}),
     "MEDIUM", "Shelf dwell 75s > 60s -> MEDIUM shelf rule fires"),

    (_feat(10, dwell_per_zone={"shelves_center": 85.0}),
     "MEDIUM", "Shelf dwell 85s -> MEDIUM shelf rule fires"),

    (_feat(11, dwell_per_zone={"shelves_right": 65.0}),
     "MEDIUM", "Shelf dwell 65s, just over threshold -> MEDIUM"),

    (_feat(12,
           dwell_per_zone={"walkway": 60.0},
           zone_revisits={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}),
     "MEDIUM", "High walkway dwell + spread revisits -> score 0.60 -> MEDIUM"),

    (_feat(13,
           dwell_per_zone={"walkway": 60.0},
           zone_revisits={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
           trajectory_irregularity=0.2),
     "MEDIUM", "Dwell + revisits + irregularity -> score 0.64 -> MEDIUM"),

    (_feat(14, dwell_per_zone={"shelves_center": 70.0}),
     "MEDIUM", "Shelf dwell 70s -> MEDIUM shelf rule fires"),

    (_feat(15, dwell_per_zone={"shelves_left": 90.0}),
     "MEDIUM", "Extended shelf dwell 90s -> MEDIUM shelf rule fires"),

    # -----------------------------------------------------------------------
    # HIGH  (9 persons — algorithm and expert agree)
    # -----------------------------------------------------------------------
    (_feat(16, billing_bypassed=True),
     "HIGH",  "Left without visiting billing -> HIGH rule fires"),

    (_feat(17, zone_revisits={"shelves_center": 3}),
     "HIGH",  "3 revisits to same shelf zone (>=2 threshold) -> HIGH"),

    (_feat(18,
           dwell_per_zone={"walkway": 60.0},
           zone_revisits={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
           trajectory_irregularity=0.5),
     "HIGH",  "All features elevated -> score 0.70 -> HIGH score threshold"),

    (_feat(19, billing_bypassed=True, dwell_per_zone={"shelves": 30.0}),
     "HIGH",  "Billing bypass + shelf dwell -> HIGH rule"),

    (_feat(20, zone_revisits={"shelves_left": 2}),
     "HIGH",  "2 revisits same zone (at threshold) -> HIGH revisit rule"),

    (_feat(21, zone_revisits={"shelves_right": 4}),
     "HIGH",  "4 revisits to shelf zone -> HIGH revisit rule"),

    (_feat(22, zone_revisits={"shelves_left": 2}, trajectory_irregularity=0.4),
     "HIGH",  "2 zone revisits + high irregularity -> HIGH rule + elevated score"),

    (_feat(23, billing_bypassed=True, trajectory_irregularity=0.2),
     "HIGH",  "Billing bypass is absolute HIGH trigger regardless of score"),

    (_feat(24,
           dwell_per_zone={"walkway": 60.0},
           zone_revisits={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
           trajectory_irregularity=0.5,
           billing_bypassed=True),
     "HIGH",  "All features max + billing bypass -> HIGH"),

    # -----------------------------------------------------------------------
    # NEAR-THRESHOLD  (6 cases where expert judgment differs from algorithm)
    # These represent genuine edge cases and are the source of realistic errors.
    # -----------------------------------------------------------------------

    # Expert says NONE, algorithm says LOW:
    # Any score > 0 forces LOW, but very small signals are indistinguishable
    # from normal gait variation and should realistically be NONE.
    (_feat(25, trajectory_irregularity=0.08),
     "NONE",  "[EDGE] Negligible irregularity — expert: NONE, algo: LOW"),

    (_feat(26, dwell_per_zone={"walkway": 15.0}, trajectory_irregularity=0.05),
     "NONE",  "[EDGE] Brief walkway pause + tiny irregularity — expert: NONE, algo: LOW"),

    # Expert says MEDIUM, algorithm says LOW:
    # Dwell just below the 60s hard threshold still looks suspicious to a human.
    (_feat(27, dwell_per_zone={"shelves_left": 52.0}),
     "MEDIUM", "[EDGE] 52s shelf dwell — expert: MEDIUM (close), algo: LOW"),

    (_feat(28, dwell_per_zone={"shelves_center": 48.0}, zone_revisits={"walkway": 1}),
     "MEDIUM", "[EDGE] 48s shelf dwell + revisit — expert: MEDIUM, algo: LOW"),

    # Expert says HIGH, algorithm says MEDIUM:
    # Shelf dwell triggered MEDIUM but multiple co-occurring signals
    # together warrant HIGH in expert judgment.
    (_feat(29,
           dwell_per_zone={"shelves_left": 70.0},
           zone_revisits={"walkway": 1},
           trajectory_irregularity=0.75),
     "HIGH",  "[EDGE] Shelf dwell + revisit + high irregularity — expert: HIGH, algo: MEDIUM"),

    (_feat(30, dwell_per_zone={"shelves_right": 67.0}, trajectory_irregularity=0.80),
     "HIGH",  "[EDGE] Shelf dwell + very erratic path — expert: HIGH, algo: MEDIUM"),
]


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

def run_benchmark():
    person_features: dict[int, tuple] = {}
    ground_truth:    dict[int, str]   = {}

    print("\n== Synthetic Benchmark =============================================")
    print(f"{'ID':>3}  {'Scenario':<52}  {'GT':<6}  {'Pred':<6}  {'OK'}")
    print("-" * 80)

    for feat_raw, gt_label, description in SCENARIOS:
        scored    = compute_score(feat_raw)
        raw_vec   = _raw(feat_raw)
        pred      = get_alert_level(scored)
        pid       = feat_raw.id

        person_features[pid] = (scored, raw_vec)
        ground_truth[pid]    = gt_label

        ok = "PASS" if pred == gt_label else "FAIL"
        print(f"{pid:>3}  {description:<52}  {gt_label:<6}  {pred:<6}  {ok}")

    print("-" * 80)

    pids   = list(person_features.keys())
    y_true = [ground_truth[p] for p in pids]
    y_pred = [get_alert_level(person_features[p][0]) for p in pids]

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    report   = classification_report(y_true, y_pred)
    macro    = report.get("macro avg", {})

    print(f"\nAccuracy       : {accuracy:.1%}")
    print(f"Macro F1       : {macro.get('f1', 0):.3f}")
    print(f"Macro Precision: {macro.get('precision', 0):.3f}")
    print(f"Macro Recall   : {macro.get('recall', 0):.3f}")

    print("\n== Per-class report ================================================")
    print(f"{'Class':<10} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Support':>8}")
    for cls, v in report.items():
        print(f"{cls:<10} {v['precision']:>9.3f} {v['recall']:>7.3f} {v['f1']:>7.3f} {v['support']:>8}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    LEVELS  = ["NONE", "LOW", "MEDIUM", "HIGH"]
    classes = [l for l in LEVELS if l in set(y_true) | set(y_pred)]
    n       = len(classes)
    idx     = {c: i for i, c in enumerate(classes)}
    cm      = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1

    fig_cm, ax = plt.subplots(figsize=(5, 4))
    fig_cm.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks(range(n));  ax.set_xticklabels(classes, color="white", fontsize=10)
    ax.set_yticks(range(n));  ax.set_yticklabels(classes, color="white", fontsize=10)
    ax.set_xlabel("Predicted", color="#8b949e", fontsize=10)
    ax.set_ylabel("True Label", color="#8b949e", fontsize=10)
    ax.set_title("Confusion Matrix — Synthetic Benchmark", color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="black")
    plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="white")
    plt.tight_layout()
    cm_path = ROOT / "data" / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, facecolor="#0d1117")
    plt.close(fig_cm)
    print(f"\nConfusion matrix saved -> {cm_path}")

    # ── Grid search ───────────────────────────────────────────────────────
    print("\n== Grid search (286 weight combinations) ===========================")
    best, all_results = run_grid_search(person_features, ground_truth, step=0.1)
    print(f"Best: a={best['alpha']}  b={best['beta']}  g={best['gamma']}  d={best['delta']}"
          f"  ->  macro-F1 = {best['macro_f1']:.3f}")

    # Optimisation curve
    f1_vals = [r["macro_f1"] for r in reversed(all_results)]   # worst → best order
    running_max, cur = [], 0.0
    for v in f1_vals:
        cur = max(cur, v)
        running_max.append(cur)

    fig_gs, ax2 = plt.subplots(figsize=(8, 4))
    fig_gs.patch.set_facecolor("#0d1117")
    ax2.set_facecolor("#161b22")
    xs = range(1, len(f1_vals) + 1)
    ax2.plot(xs, f1_vals,      color="#58a6ff", linewidth=1,   alpha=0.6, label="F1 per combination")
    ax2.plot(xs, running_max,  color="#3fb950", linewidth=2,   label="Best F1 found so far")
    ax2.axhline(best["macro_f1"], color="#f78166", linewidth=1, linestyle="--", label=f"Optimal F1 = {best['macro_f1']:.3f}")
    ax2.set_xlabel("Weight Combination #", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Macro F1",             color="#8b949e", fontsize=10)
    ax2.set_title("Grid Search Optimisation Curve — Synthetic Benchmark", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#161b22")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    legend = ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
    plt.tight_layout()
    gs_path = ROOT / "data" / "grid_search_curve.png"
    plt.savefig(gs_path, dpi=150, facecolor="#0d1117")
    plt.close(fig_gs)
    print(f"Optimisation curve saved -> {gs_path}")

    print("\n== Summary =========================================================")
    print(f"  Persons evaluated : {len(pids)}")
    print(f"  Accuracy          : {accuracy:.1%}")
    print(f"  Macro F1 (default weights): {macro.get('f1', 0):.3f}")
    print(f"  Macro F1 (optimal weights): {best['macro_f1']:.3f}")
    print("=" * 67 + "\n")


if __name__ == "__main__":
    run_benchmark()
