"""
SHAP-based feature attribution for the linear risk scoring model.

The scoring model is a linear combination:
    S = ALPHA*f1 + BETA*f2 + GAMMA*f3 + DELTA*f4

For linear models, SHAP values reduce analytically to:
    phi_i = w_i * (x_i - E[x_i])     (contribution of feature i)
    phi_0 = w · E[x]                   (population baseline)

SHAPExplainer collects all observed feature vectors as the pipeline runs,
then computes attributions for every tracked person relative to the
population average (rather than an arbitrary fixed baseline).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from src.behavior.scoring import ALPHA, BETA, GAMMA, DELTA

_WEIGHTS = np.array([ALPHA, BETA, GAMMA, DELTA], dtype=float)
FEATURE_NAMES = ["Dwell Anomaly", "Zone Revisits", "Path Irregularity", "Billing Bypass"]

_COL_POS = "#ef4444"   # red  — feature pushes risk above baseline
_COL_NEG = "#22c55e"   # green — feature pulls risk below baseline
_COL_BASE = "#388bfd"  # blue  — baseline bar


class SHAPExplainer:
    """
    Collects (f1, f2, f3, f4) feature vectors as the pipeline runs and
    computes SHAP attributions for every tracked person using the
    observed population as the background distribution.

    Usage:
        explainer = SHAPExplainer()
        # inside per-frame loop:
        explainer.update(pid, breakdown["raw"])
        # after processing:
        explanations = explainer.explain_all()
    """

    def __init__(self) -> None:
        self._all_vectors: list[list[float]] = []
        self._person_vectors: dict[int, list[float]] = {}

    def update(self, pid: int, raw_features: list[float]) -> None:
        """Register the latest normalized feature vector for a tracked person."""
        self._all_vectors.append(list(raw_features))
        self._person_vectors[pid] = list(raw_features)

    def n_samples(self) -> int:
        return len(self._all_vectors)

    def explain_all(self) -> dict[int, dict]:
        """
        Compute SHAP attributions for every tracked person.

        Uses shap.LinearExplainer when the library is available; falls back to
        the identical analytical formula (valid for any linear model).

        Returns dict mapping pid -> explanation dict with keys:
            shap_values, feature_values, base_value, feature_names, prediction
        """
        if len(self._all_vectors) < 2:
            return {}

        bg = np.array(self._all_vectors, dtype=float)
        expected = bg.mean(axis=0)
        base_value = float(_WEIGHTS @ expected)

        results: dict[int, dict] = {}

        if _HAS_SHAP:
            # Use the official SHAP library for correctness / auditability.
            # (coef, intercept) tuple form accepted by LinearExplainer.
            explainer = _shap.LinearExplainer((_WEIGHTS, 0.0), bg)

        for pid, fvec in self._person_vectors.items():
            x = np.array(fvec, dtype=float)

            if _HAS_SHAP:
                sv = explainer.shap_values(x.reshape(1, -1))[0]
                ev = float(explainer.expected_value)
            else:
                # Analytical formula: phi_i = w_i * (x_i - E[x_i])
                sv = _WEIGHTS * (x - expected)
                ev = base_value

            results[pid] = {
                "shap_values":    sv,
                "feature_values": x,
                "base_value":     ev,
                "feature_names":  FEATURE_NAMES,
                "prediction":     float(np.clip(_WEIGHTS @ x, 0.0, 1.0)),
            }

        return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_waterfall(explanation: dict, pid: int) -> plt.Figure:
    """
    Horizontal waterfall chart showing per-feature SHAP contributions.

    Red bars push the risk score above the population baseline;
    green bars pull it below.
    """
    sv   = explanation["shap_values"]
    fv   = explanation["feature_values"]
    names = explanation["feature_names"]
    base = explanation["base_value"]
    pred = explanation["prediction"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = [_COL_POS if v >= 0 else _COL_NEG for v in sv]
    y_labels = [f"{n}\n(={fv[i]:.2f})" for i, n in enumerate(names)]

    bars = ax.barh(y_labels, sv, color=colors, height=0.52, edgecolor="none")
    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, sv):
        pad = 0.003
        ha = "left" if val >= 0 else "right"
        x_pos = (bar.get_x() + bar.get_width() + pad) if val >= 0 else (bar.get_x() - pad)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8, color="#e6edf3")

    ax.set_title(
        f"Person {pid}  |  Risk score: {pred:.3f}  (pop. baseline: {base:.3f})",
        color="#e6edf3", fontsize=9, pad=7,
    )
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("SHAP value  (contribution to risk score vs. population average)",
                  color="#8b949e", fontsize=7.5)

    # Annotate baseline
    ax.axvline(base - base, color="none")   # ensure 0 is always visible
    plt.tight_layout(pad=0.8)
    return fig


def plot_summary_bar(explanations: dict[int, dict]) -> plt.Figure:
    """
    Stacked bar chart: mean |SHAP| per feature across all tracked persons.
    Gives an overall view of which features drive risk in this video.
    """
    if not explanations:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return fig

    all_sv = np.array([e["shap_values"] for e in explanations.values()])
    mean_abs = np.abs(all_sv).mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    palette = ["#388bfd", "#d29922", "#ef4444", "#3fb950"]
    bars = ax.barh(FEATURE_NAMES, mean_abs, color=palette, height=0.5, edgecolor="none")

    for bar, val in zip(bars, mean_abs):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color="#e6edf3")

    ax.set_title("Mean |SHAP| per feature  (overall risk drivers for this video)",
                 color="#e6edf3", fontsize=9, pad=7)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Mean absolute SHAP value", color="#8b949e", fontsize=7.5)
    plt.tight_layout(pad=0.8)
    return fig
