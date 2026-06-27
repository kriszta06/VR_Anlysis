import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ground_truth_handler import load_ground_truth
from src.config import disability_config as config

HEALTHY_COLOR  = '#22C55E'
DISABLED_COLOR = '#EF4444'
BG             = '#F8FAFC'


def compute_mann_whitney(scores_healthy, scores_disabled):
    """
    Performs the Mann–Whitney U test to compare behavioral variation scores
    between healthy and disabled participant groups and estimates the associated
    effect size.

    The function computes both two-sided and one-sided Mann–Whitney U tests,
    calculates the rank-biserial correlation as a measure of effect size,
    estimates a 95% confidence interval for the effect size using bootstrap
    resampling, evaluates the probability of superiority, and generates
    descriptive statistics for both groups.

    Parameters
    ----------
    scores_healthy : array-like
        Behavioral variation scores for participants in the healthy group.

    scores_disabled : array-like
        Behavioral variation scores for participants in the disabled group.

    Returns
    -------
    dict
        Dictionary containing the statistical test results and descriptive
        statistics, including:

        - ``n_healthy`` : int
            Number of healthy participants.
        - ``n_disabled`` : int
            Number of disabled participants.
        - ``u_statistic`` : float
            Mann–Whitney U statistic.
        - ``u_max`` : float
            Maximum possible U statistic.
        - ``p_value_two_sided`` : float
            Two-sided p-value.
        - ``p_value_one_sided`` : float
            One-sided p-value (alternative hypothesis: disabled > healthy).
        - ``effect_size_r`` : float
            Rank-biserial correlation.
        - ``effect_size_ci_95`` : list of float
            Bootstrap 95% confidence interval for the effect size.
        - ``probability_superiority`` : float
            Probability that a randomly selected disabled participant has a
            higher score than a randomly selected healthy participant.
        - ``direction`` : str
            Direction of the observed effect.
        - ``effect_label`` : str
            Qualitative interpretation of the effect size.
        - ``effect_description`` : str
            Textual interpretation of the observed effect.
        - ``significance`` : str
            Interpretation of the statistical significance.
        - ``reject_h0_alpha05`` : bool
            Indicates whether the null hypothesis is rejected at α = 0.05.
        - ``reject_h0_alpha10`` : bool
            Indicates whether the null hypothesis is rejected at α = 0.10.
        - Group summary statistics (mean, median, and standard deviation).
        - ``boot_r_distribution`` : list of float
            Sample of bootstrap effect size estimates.

    Notes
    -----
    - The null hypothesis assumes identical score distributions for the healthy
    and disabled groups.
    - The one-sided alternative hypothesis assumes that the disabled group
    exhibits systematically higher behavioral variation scores than the
    healthy group.
    - Rank-biserial correlation is used as the effect size measure.
    - A non-parametric bootstrap procedure with 2,000 resampling iterations is
    employed to estimate the 95% confidence interval of the effect size.
    - The returned bootstrap distribution is truncated to the first 200 samples
    to reduce the size of the serialized JSON output.
    """
    n1 = len(scores_healthy)
    n2 = len(scores_disabled)

    u_stat, p_two_sided = stats.mannwhitneyu(
        scores_disabled, scores_healthy,
        alternative='two-sided'
    )
    _, p_one_sided = stats.mannwhitneyu(
        scores_disabled, scores_healthy,
        alternative='greater'
    )

    r_raw = 1 - (2 * u_stat) / (n1 * n2)
    r = abs(r_raw)
    direction = "atypical > typical" if r_raw < 0 else "typical > atypical"

    prob_superiority = u_stat / (n1 * n2)

    np.random.seed(42)
    n_boot = 2000
    boot_r = []
    for _ in range(n_boot):
        s_h = np.random.choice(scores_healthy,  size=n1, replace=True)
        s_d = np.random.choice(scores_disabled, size=n2, replace=True)
        u_b, _ = stats.mannwhitneyu(s_d, s_h, alternative='two-sided')
        boot_r.append(abs(1 - (2 * u_b) / (n1 * n2)))

    ci_low  = float(np.percentile(boot_r, 2.5))
    ci_high = float(np.percentile(boot_r, 97.5))

    abs_r = abs(r)
    if abs_r >= 0.50:
        effect_label = "MAJOR"
        effect_desc  = "Difference between groups is major"
    elif abs_r >= 0.30:
        effect_label = "MEDIUM"
        effect_desc  = "Difference between groups is medium"
    else:
        effect_label = "MINOR"
        effect_desc  = "Difference between groups is minor"

    # Interpretare p-value
    if p_two_sided < 0.01:
        significance = "Highly significant (p < 0.01)"
    elif p_two_sided < 0.05:
        significance = "Significant (p < 0.05)"
    elif p_two_sided < 0.10:
        significance = "Lowly significant (p < 0.10)"
    else:
        significance = "Not significant (p ≥ 0.10)"

    return {
        "n_healthy": n1,
        "n_disabled": n2,
        "u_statistic": float(u_stat),
        "u_max": float(n1 * n2),
        "p_value_two_sided": float(p_two_sided),
        "p_value_one_sided": float(p_one_sided),
        "effect_size_r": float(r),
        "effect_size_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "probability_superiority": float(prob_superiority),
        "direction": direction,
        "effect_label": effect_label,
        "effect_description": effect_desc,
        "significance": significance,
        "reject_h0_alpha05": bool(p_two_sided < 0.05),
        "reject_h0_alpha10": bool(p_two_sided < 0.10),
        "median_healthy": float(np.median(scores_healthy)),
        "median_disabled": float(np.median(scores_disabled)),
        "mean_healthy": float(np.mean(scores_healthy)),
        "mean_disabled": float(np.mean(scores_disabled)),
        "std_healthy": float(np.std(scores_healthy)),
        "std_disabled": float(np.std(scores_disabled)),
        "boot_r_distribution": [round(float(x), 5) for x in boot_r[:200]],  
    }


def generate_plots(per_person, results, output_dir=None):
    """
    Generates a multi-panel statistical figure summarising the Mann–Whitney U test
    results comparing behavioural atypicality scores between healthy and
    motor-impaired participant groups.

    The function produces a 2×3 grid layout saved as a PNG file, comprising a
    violin and strip plot with annotated medians and significance bracket, a
    summary card reporting key inferential statistics, a bootstrap histogram
    of the effect size distribution with conventional Cohen thresholds overlaid,
    and a plain-language interpretability panel expressing the probability of
    superiority relative to the chance-level baseline.

    Parameters
    ----------
    per_person : list of dict
        Per-participant records, each containing:

        - ``person`` : str
            Participant identifier (e.g. ``"Person_01"``).
        - ``final_score`` : float
            Composite behavioural atypicality score in the range [0, 1].
        - ``true_label`` : int
            Group membership: ``0`` = healthy control, ``1`` = motor-impaired.

    results : dict
        Aggregate statistical outputs returned by the Mann–Whitney U test
        procedure, including:

        - ``u_statistic`` : float
            Mann–Whitney U statistic.
        - ``u_max`` : float
            Maximum possible U value (n₁ × n₂).
        - ``p_value_two_sided`` : float
            Two-sided p-value.
        - ``p_value_one_sided`` : float
            One-sided p-value.
        - ``effect_size_r`` : float
            Rank-biserial correlation coefficient.
        - ``effect_size_ci_95`` : list of float
            Bootstrap 95% confidence interval for the effect size.
        - ``probability_superiority`` : float
            Probability that a randomly selected motor-impaired participant
            scores higher than a randomly selected healthy control.
        - ``effect_label`` : str
            Qualitative interpretation of the effect magnitude.
        - ``median_healthy`` : float
            Median behavioural atypicality score for the healthy group.
        - ``median_disabled`` : float
            Median behavioural atypicality score for the motor-impaired group.
        - ``boot_r_distribution`` : list of float
            Sample of bootstrap effect size estimates used for histogram rendering.

    output_dir : str, optional
        Path to the directory where the output figure is saved. Created
        recursively if it does not exist.
        Defaults to ``"data/output/evaluation_results"``.

    Returns
    -------
    str
        Path to the saved figure file (``<output_dir>/mann_whitney_plot.png``).

    Notes
    -----
    - The function relies on module-level colour constants ``BG``,
    ``HEALTHY_COLOR``, and ``DISABLED_COLOR`` for consistent cross-figure
    theming.
    - A fixed random seed (``np.random.seed(3)``) is applied prior to jittering
    strip-plot markers to ensure reproducible figure layouts across runs.
    - The significance bracket annotation follows the conventional scheme:
    ``*`` for p < 0.05, ``†`` for p < 0.10, and ``ns`` otherwise.
    - Effect size thresholds follow Cohen's conventions for rank-biserial
    correlation: |r| ≥ 0.50 = large, 0.30 ≤ |r| < 0.50 = medium,
    |r| < 0.30 = small.
    - The figure is exported at 170 DPI with ``bbox_inches='tight'`` to prevent
    label clipping and is explicitly closed after saving to release memory,
    making the function safe to invoke within batch evaluation pipelines.
    """

    if output_dir is None:
        output_dir = BASE_DIR / "data" / "output" / "evaluation_results"
    else:
        output_dir = BASE_DIR / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scores_h = np.array([p["final_score"] for p in per_person if p["true_label"] == 0])
    scores_d = np.array([p["final_score"] for p in per_person if p["true_label"] == 1])
    names_h  = [p["person"].replace("Person_", "P") for p in per_person if p["true_label"] == 0]
    names_d  = [p["person"].replace("Person_", "P") for p in per_person if p["true_label"] == 1]

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(BG)

    vp = ax1.violinplot([scores_h, scores_d], positions=[0, 1],
                        showmedians=True, showextrema=True, widths=0.6)
    vp['bodies'][0].set_facecolor(HEALTHY_COLOR);  vp['bodies'][0].set_alpha(0.35)
    vp['bodies'][1].set_facecolor(DISABLED_COLOR); vp['bodies'][1].set_alpha(0.35)
    for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if part in vp:
            vp[part].set_color(['#15803D', '#B91C1C'])
            vp[part].set_linewidth(2)

    np.random.seed(3)
    for i, (score, name) in enumerate(zip(scores_h, names_h)):
        jit = np.random.uniform(-0.08, 0.08)
        ax1.scatter(0 + jit, score, color=HEALTHY_COLOR, s=70,
                    zorder=4, alpha=0.85, edgecolors='white', linewidths=0.6)
        ax1.text(0 + jit, score + 0.022, name, ha='center',
                 fontsize=7, color='#15803D', fontweight='bold')

    for i, (score, name) in enumerate(zip(scores_d, names_d)):
        jit = np.random.uniform(-0.08, 0.08)
        ax1.scatter(1 + jit, score, color=DISABLED_COLOR, s=100,
                    zorder=4, alpha=0.9, edgecolors='white', linewidths=0.6, marker='D')
        ax1.text(1 + jit, score + 0.022, name, ha='center',
                 fontsize=7, color='#B91C1C', fontweight='bold')

    ax1.plot([-0.15, 0.15], [results["median_healthy"],  results["median_healthy"]],
             color='#15803D', linewidth=2.5, linestyle='-', zorder=5)
    ax1.plot([0.85,  1.15], [results["median_disabled"], results["median_disabled"]],
             color='#B91C1C', linewidth=2.5, linestyle='-', zorder=5)
    ax1.text(0.22, results["median_healthy"],
             f'Median\n{results["median_healthy"]:.3f}',
             color='#15803D', fontsize=8, va='center')
    ax1.text(1.22, results["median_disabled"],
             f'Median\n{results["median_disabled"]:.3f}',
             color='#B91C1C', fontsize=8, va='center')

    y_bracket = max(scores_h.max(), scores_d.max()) + 0.05
    ax1.plot([0, 0, 1, 1], [y_bracket, y_bracket + 0.02,
                              y_bracket + 0.02, y_bracket],
             color='#374151', linewidth=1.5)
    p_val = results["p_value_two_sided"]
    p_txt = f'p = {p_val:.4f}' + (' *' if p_val < 0.05 else (' †' if p_val < 0.10 else ' ns'))
    ax1.text(0.5, y_bracket + 0.03, p_txt, ha='center', fontsize=10,
             color='#374151', fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Healthy Controls  (n=14)', 'Motor-Impaired  (n=5)'], fontsize=11)
    ax1.set_ylabel('Behavioural Atypicality Score', fontsize=10)
    ax1.set_title('Score Distribution by Group — Mann-Whitney U Test', fontsize=11, fontweight='bold')
    ax1.set_xlim(-0.55, 1.8)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')
    ax1.spines[['top', 'right']].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('white')
    ax2.axis('off')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    p_val  = results["p_value_two_sided"]
    r_val  = results["effect_size_r"]
    ci     = results["effect_size_ci_95"]
    ps     = results["probability_superiority"]

    color_p = '#15803D' if p_val < 0.05 else ('#92400E' if p_val < 0.10 else '#B91C1C')
    color_r = '#15803D' if abs(r_val) > 0.50 else ('#92400E' if abs(r_val) > 0.30 else '#B91C1C')

    lines = [
        (0.92, 'TEST RESULTS', '#1F2937', 11, 'bold'),
        (0.92, '─────────────────────', '#D1D5DB', 8, 'normal'),
        (0.78, f'U = {results["u_statistic"]:.0f}  /  {results["u_max"]:.0f} max', '#374151', 10, 'normal'),
        (0.92, '', '#374151', 9, 'normal'),
    ]

    ax2.text(0.5, 0.97, 'TEST RESULTS', ha='center', fontsize=11,
             fontweight='bold', color='#1F2937', transform=ax2.transAxes)
    ax2.plot([0.05, 0.95], [0.93, 0.93], color='#D1D5DB', linewidth=1,
             transform=ax2.transAxes)

    entries = [
        ('U Statistic',f'{results["u_statistic"]:.0f} / {results["u_max"]:.0f}', '#374151'),
        ('p-value (two-sided)',f'{p_val:.4f}',color_p),
        ('p-value (one-sided)',f'{results["p_value_one_sided"]:.4f}',color_p),
        ('Effect size r',f'{r_val:.3f}',color_r),
        ('95% CI for r',f'[{ci[0]:.3f}, {ci[1]:.3f}]',color_r),
        ('P(superiority)',f'{ps:.1%}',color_r),
        ('Effect magnitude',results["effect_label"],color_r),
    ]

    for idx, (lbl, val, col) in enumerate(entries):
        y = 0.85 - idx * 0.115
        ax2.text(0.05, y, lbl + ':', fontsize=8.5, color='#6B7280',
                 transform=ax2.transAxes, va='center')
        ax2.text(0.97, y, val, fontsize=9, color=col, fontweight='bold',
                 transform=ax2.transAxes, va='center', ha='right')

    sig_color = '#15803D' if p_val < 0.05 else '#92400E'
    sig_label = 'Significant\n  statistic (α=0.05)' if p_val < 0.05 else \
                'Marginal\n  semnificativ (α=0.10)' if p_val < 0.10 else \
                'Not significant\n  (α=0.05)'
    rect = plt.Rectangle((0.05, 0.02), 0.90, 0.10, fill=True,
                          facecolor=sig_color + '22', edgecolor=sig_color,
                          linewidth=1.5, transform=ax2.transAxes, clip_on=False)
    ax2.add_patch(rect)
    ax2.text(0.5, 0.07, sig_label, ha='center', va='center', fontsize=9,
             color=sig_color, fontweight='bold', transform=ax2.transAxes)

    ax2.set_title('Test statistics', fontsize=10, fontweight='bold', pad=6)

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(BG)

    boot_vals = np.array(results["boot_r_distribution"])
    ax3.hist(boot_vals, bins=35, color='#2563EB', alpha=0.55, edgecolor='white',
             linewidth=0.5, label='Bootstrap distribution (n=200 din 2000)')
    ax3.axvline(r_val, color='#DC2626', linewidth=2.5, linestyle='-',
                label=f'Effect size observed r = {r_val:.3f}')
    ax3.axvline(ci[0], color='#F59E0B', linewidth=1.8, linestyle='--',
                label=f'CI 95%: [{ci[0]:.3f}, {ci[1]:.3f}]')
    ax3.axvline(ci[1], color='#F59E0B', linewidth=1.8, linestyle='--')
    ax3.axvspan(ci[0], ci[1], alpha=0.10, color='#F59E0B')

    for val, lbl, col in [(0.30, 'efect mediu', '#6B7280'), (0.50, 'efect mare', '#374151')]:
        ax3.axvline(val, color=col, linewidth=1, linestyle=':', alpha=0.7)
        ax3.text(val + 0.005, ax3.get_ylim()[1] * 0.85 if ax3.get_ylim()[1] > 0 else 5,
                 lbl, fontsize=7.5, color=col, style='italic')

    ax3.set_xlabel('Rank-biserial r (effect size)', fontsize=10)
    ax3.set_ylabel('Bootstrap frequencies', fontsize=10)
    ax3.set_title('Bootstrap distribution of the effect size — stability of the estimate', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8.5, framealpha=0.9)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.grid(axis='y', alpha=0.25, linestyle='--')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('white')
    ax4.axis('off')
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

    ax4.text(0.5, 0.95, 'WHAT DOES IT MEAN', ha='center', fontsize=10,
             fontweight='bold', color='#1F2937', transform=ax4.transAxes)
    ax4.text(0.5, 0.88, f'r = {r_val:.3f}?', ha='center', fontsize=14,
             fontweight='bold', color=color_r, transform=ax4.transAxes)
    ax4.plot([0.05,0.95],[0.83,0.83],color='#D1D5DB',linewidth=1,transform=ax4.transAxes)

    explanation = (
        f"If you randomly select\n"
        f"one disabled participant\n"
        f"and one healthy participant,\n\n"
        f"the probability that the\n"
        f"disabled participant has a\n"
        f"HIGHER score is:\n\n"
    )
    ax4.text(0.5, 0.55, explanation, ha='center', va='center',
             fontsize=9, color='#374151', transform=ax4.transAxes,
             linespacing=1.5)

    ax4.text(0.5, 0.22, f'{ps:.1%}', ha='center', va='center',
             fontsize=28, fontweight='bold', color=color_r,
             transform=ax4.transAxes)

    ax4.text(0.5, 0.08, f'(compared to 50% in distribution\nindex — H0)', ha='center',
             fontsize=7.5, color='#9CA3AF', transform=ax4.transAxes, style='italic')

    ax4.set_title('Intuitive interpretation', fontsize=10, fontweight='bold', pad=6)

    fig.text(0.5, 0.01,
            '* p < 0.05  † p < 0.10  ns = not significant  '
            '|r| > 0.50 = large effect  |r| 0.30–0.50 = medium effect  |r| < 0.30 = small effect',
            ha='center', fontsize=7.5, color='#9CA3AF', style='italic')

    plt.suptitle('Mann-Whitney U Test — Statistical Validation\n'
                 'Corelation between disability and motor impairment',
                 fontsize=13, fontweight='bold', y=1.01)

    out_path = Path(output_dir) / "mann_whitney_plot.png"
    plt.savefig(str(out_path), dpi=170, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Graphic saved: {out_path}")
    return str(out_path)


def print_summary(results):
    """
    Prints a formatted summary of the Mann–Whitney U test results to standard
    output, including descriptive statistics, inferential test outcomes, effect
    size estimates, and a hypothesis rejection verdict.

    The summary is structured into three sections: group-level descriptive
    statistics, inferential test statistics with effect size and confidence
    interval, and a plain-language conclusion with an explicit statement on
    null hypothesis rejection at both α = 0.05 and α = 0.10 significance
    levels.

    Parameters
    ----------
    results : dict
        Dictionary of statistical outputs as returned by the Mann–Whitney U
        test procedure, containing:

        - ``n_healthy`` : int
            Number of healthy control participants.
        - ``n_disabled`` : int
            Number of motor-impaired participants.
        - ``median_healthy`` : float
            Median behavioural atypicality score for the healthy group.
        - ``median_disabled`` : float
            Median behavioural atypicality score for the motor-impaired group.
        - ``std_healthy`` : float
            Standard deviation of scores for the healthy group.
        - ``std_disabled`` : float
            Standard deviation of scores for the motor-impaired group.
        - ``u_statistic`` : float
            Mann–Whitney U statistic.
        - ``u_max`` : float
            Maximum possible U value (n₁ × n₂).
        - ``p_value_two_sided`` : float
            Two-sided p-value.
        - ``p_value_one_sided`` : float
            One-sided p-value.
        - ``effect_size_r`` : float
            Rank-biserial correlation coefficient.
        - ``effect_label`` : str
            Qualitative interpretation of the effect magnitude.
        - ``effect_size_ci_95`` : list of float
            Bootstrap 95% confidence interval for the effect size.
        - ``probability_superiority`` : float
            Probability that a randomly selected motor-impaired participant
            scores higher than a randomly selected healthy control.
        - ``significance`` : str
            Textual interpretation of the statistical significance.
        - ``effect_description`` : str
            Plain-language description of the observed effect.
        - ``reject_h0_alpha05`` : bool
            Whether the null hypothesis is rejected at α = 0.05.
        - ``reject_h0_alpha10`` : bool
            Whether the null hypothesis is rejected at α = 0.10.

    Returns
    -------
    None
        Outputs are written exclusively to standard output.

    Notes
    -----
    - When the null hypothesis cannot be rejected at either threshold, the
      function explicitly attributes the lack of significance to limited
      statistical power arising from the small motor-impaired sample size
      (n = 5), rather than to the absence of a true effect.
    - The verdict block is mutually exclusive: only the most stringent
      applicable threshold is reported.
    """
    print(f"\n{'='*60}")
    print("  MANN-WHITNEY U TEST — RESULTS")
    print(f"{'='*60}")
    print(f"  Healthy controls (n={results['n_healthy']}): median = {results['median_healthy']:.4f}"
          f" {results['std_healthy']:.4f}")
    print(f" Motor-impaired (n={results['n_disabled']}): median = {results['median_disabled']:.4f}"
          f" {results['std_disabled']:.4f}")
    print(f"\nU statistic: {results['u_statistic']:.0f} (max possible: {results['u_max']:.0f})")
    print(f"p-value (two-sided): {results['p_value_two_sided']:.4f}")
    print(f"p-value (one-sided): {results['p_value_one_sided']:.4f}")
    print(f"\nEffect size r: {results['effect_size_r']:.3f} ({results['effect_label']})")
    print(f"95% CI: [{results['effect_size_ci_95'][0]:.3f}, {results['effect_size_ci_95'][1]:.3f}]")
    print(f"P(superiority): {results['probability_superiority']:.1%}")
    print(f"\nConclusion: {results['significance']}")
    print(f"{results['effect_description']}")
    print(f"{'='*60}")

    if results['reject_h0_alpha05']:
        print("\nH0 rejected at α = 0.05: scores differ significantly between groups, independent of any threshold selection.")
    elif results['reject_h0_alpha10']:
        print("\nH0 rejected at α = 0.10 (marginal): a separation tendency is observed, but does not reach the conventional α = 0.05 threshold.")
    else:
        print("\nH0 cannot be rejected at α = 0.05. With only n = 5 motor-impaired participants, statistical power is limited, a true effect may be present but undetected due to small sample size.")
    print()
def run_mann_whitney(
    person_scores=None,
    ground_truth_path="data/ground_truth/ground_truth.csv",
    json_path="data/output/disability_results/behavioral_classification.json",
    output_dir=None,
):
    """
    Orchestrates the full Mann–Whitney U test evaluation pipeline, from data
    loading through statistical testing, result serialisation, and figure
    generation.

    The function supports two invocation modes:

    1. **Integrated mode** — called from the main pipeline with a pre-computed
       ``person_scores`` dictionary, bypassing JSON loading entirely.
    2. **Standalone mode** — called without arguments, reading behavioural
       scores directly from the ``behavioral_classification.json`` output file.

    In both modes the function aligns scores with ground-truth labels, separates
    participants into healthy and motor-impaired groups, runs the statistical
    test, prints a formatted summary, generates diagnostic figures, and
    serialises results to disk.

    Parameters
    ----------
    person_scores : dict of {str: float}, optional
        Mapping of participant identifiers to their final behavioural
        atypicality scores. If ``None``, scores are loaded from ``json_path``.
        Identifiers may include or omit the ``"Person_"`` prefix; normalisation
        is applied internally.

    ground_truth_path : str, optional
        Path to the CSV file containing ground-truth group labels.
        Defaults to ``"data/ground_truth/ground_truth.csv"``.

    json_path : str, optional
        Path to the JSON file produced by the behavioural classification
        pipeline. Used only when ``person_scores`` is ``None``.
        Defaults to
        ``"data/output/disability_results/behavioral_classification.json"``.

    output_dir : str, optional
        Directory where the output figure and JSON results file are saved.
        Created recursively if it does not exist.
        Defaults to ``"data/output/evaluation_results"``.

    Returns
    -------
    dict or None
        Dictionary of statistical results as returned by
        ``compute_mann_whitney``, augmented with a ``per_person_scores`` key
        containing per-participant group labels and scores. Returns ``None``
        if a required input file is missing or if fewer than two motor-impaired
        participants are available.

    Notes
    -----
    - Participant identifiers are normalised by stripping whitespace and
      removing the ``"Person_"`` prefix before matching against ground-truth
      labels, ensuring compatibility with both naming conventions.
    - The ``boot_r_distribution`` key is excluded from the serialised JSON
      output to limit file size; all other fields are converted to
      JSON-compatible Python types via ``make_serializable``.
    - The output JSON file is written with ``ensure_ascii=False`` to preserve
      any non-ASCII characters in participant identifiers.
    """

    if person_scores is None:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"ERROR: File not found at{json_path}. Run main.py first.")
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        person_scores = {}
        for pid, info in data.get("persons", {}).items():
            normalized = str(pid).strip()
            if normalized.lower().startswith("person_"):
                normalized = normalized.split("_", 1)[1]
            person_scores[normalized] = float(info["final_score"])

    ground_truth = load_ground_truth(ground_truth_path)
    if not ground_truth:
        print(f"ERROR: Ground truth file not found at {ground_truth_path}.")
        return None

    gt_normalized = {}
    for k, v in ground_truth.items():
        key = str(k).strip()
        if key.lower().startswith("person_"):
            key = key.split("_", 1)[1]
        gt_normalized[key] = int(v)

    scores_healthy  = []
    scores_disabled = []
    per_person      = []

    for pid, score in person_scores.items():
        if pid not in gt_normalized:
            continue
        label = gt_normalized[pid]
        per_person.append({"person": f"Person_{pid}", "true_label": label, "final_score": score})
        if label == 0:
            scores_healthy.append(score)
        else:
            scores_disabled.append(score)

    if len(scores_disabled) < 2:
        print("ERROR: At least 2 motor-impaired participants are required.")
        return None

    results = compute_mann_whitney(
        np.array(scores_healthy),
        np.array(scores_disabled)
    )

    print_summary(results)

    if output_dir is None:
        output_dir = BASE_DIR / "data" / "output" / "evaluation_results"
    else:
        output_dir = BASE_DIR / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_plots(per_person, results, output_dir)

    out_json = Path(output_dir) / "mann_whitney_results.json"

    def make_serializable(v):
        """
        Converts a single value to a JSON-compatible Python type.

        Handles NumPy scalar types that are not natively serialisable by the
        standard ``json`` module, mapping them to their closest built-in
        Python equivalents.

        Parameters
        ----------
        v : any
            Value to convert. Supported NumPy types are ``np.bool_``,
            ``np.integer``, and ``np.floating``. All other types are returned
            unchanged.

        Returns
        -------
        bool, int, float, or any
            The converted value if ``v`` is a recognised NumPy scalar type;
            otherwise ``v`` is returned as-is.
        """
        if isinstance(v, (bool, np.bool_)):   return bool(v)
        if isinstance(v, np.integer):          return int(v)
        if isinstance(v, np.floating):         return float(v)
        return v

    results_to_save = {k: make_serializable(v) for k, v in results.items() if k != "boot_r_distribution"}
    results_to_save["per_person_scores"] = [
        {
            "person": p["person"],
            "true_label": bool(p["true_label"]),
            "final_score": float(p["final_score"])
        }
        for p in per_person
    ]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {out_json}")

    return results

if __name__ == "__main__":
    run_mann_whitney()