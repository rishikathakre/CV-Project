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

# The model is a simple weighted sum, so the weights are the coefficients.
_WEIGHTS = np.array([ALPHA, BETA, GAMMA, DELTA], dtype=float)
FEATURE_NAMES = ["Dwell Anomaly", "Zone Revisits", "Path Irregularity", "Billing Bypass"]

_COL_POS  = "#ef4444"  # red for features that push risk up
_COL_NEG  = "#22c55e"  # green for features that push risk down
_COL_BASE = "#388bfd"  # blue for the baseline


class SHAPExplainer:
    """Explains why each person got their risk score relative to the crowd average.

    SHAP (SHapley Additive exPlanations) tells us how much each feature
    contributed to a person's score compared to the average score of everyone
    in the scene. A positive value means that feature made their score higher.

    Because our model is a linear weighted sum, the SHAP value for feature i
    is simply: weight_i * (person_value_i - crowd_mean_i).
    """

    def __init__(self) -> None:
        self._all_vectors: list[list[float]] = []       # every observation ever seen
        self._person_vectors: dict[int, list[float]] = {}  # latest vector per person

    def update(self, pid: int, raw_features: list[float]) -> None:
        """Add a new observation for one person."""
        self._all_vectors.append(list(raw_features))
        self._person_vectors[pid] = list(raw_features)

    def n_samples(self) -> int:
        return len(self._all_vectors)

    def explain_all(self) -> dict[int, dict]:
        """Compute SHAP values for every person using the full observation history as background.

        Returns an empty dict if fewer than 2 observations have been collected.
        """
        if len(self._all_vectors) < 2:
            return {}

        bg = np.array(self._all_vectors, dtype=float)
        expected = bg.mean(axis=0)
        base_value = float(_WEIGHTS @ expected)

        results: dict[int, dict] = {}

        if _HAS_SHAP:
            explainer = _shap.LinearExplainer((_WEIGHTS, 0.0), bg)

        for pid, fvec in self._person_vectors.items():
            x = np.array(fvec, dtype=float)
            if _HAS_SHAP:
                sv = explainer.shap_values(x.reshape(1, -1))[0]
                ev = float(explainer.expected_value)
            else:
                # Fallback formula: phi_i = w_i * (x_i - mean_i)
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


def plot_waterfall(explanation: dict, pid: int) -> plt.Figure:
    """Draw a horizontal bar chart showing each feature's contribution for one person."""
    sv    = explanation["shap_values"]
    fv    = explanation["feature_values"]
    names = explanation["feature_names"]
    base  = explanation["base_value"]
    pred  = explanation["prediction"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors   = [_COL_POS if v >= 0 else _COL_NEG for v in sv]
    y_labels = [f"{n}\n(={fv[i]:.2f})" for i, n in enumerate(names)]

    bars = ax.barh(y_labels, sv, color=colors, height=0.52, edgecolor="none")
    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, sv):
        pad   = 0.003
        ha    = "left" if val >= 0 else "right"
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
    ax.axvline(base - base, color="none")
    plt.tight_layout(pad=0.8)
    return fig


def plot_summary_bar(explanations: dict[int, dict]) -> plt.Figure:
    """Draw a bar chart of mean absolute SHAP values across all tracked people.

    This shows which feature drives risk the most in the current scene overall.
    """
    if not explanations:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return fig

    all_sv   = np.array([e["shap_values"] for e in explanations.values()])
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
