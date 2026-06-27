import numpy as np
import json, os
import itertools
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

def load_ground_truth(csv_path="data/ground_truth/ground_truth.csv"):
    """
    Loads ground-truth group labels from a CSV file.

    The file is expected to contain two columns — ``person_id`` and
    ``diagnosis`` — with no header row required, although a header is
    tolerated and skipped automatically if detected. Labels must be
    integer values (``0`` = healthy control, ``1`` = motor-impaired).

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the ground-truth CSV file.
        Defaults to ``"data/ground_truth/ground_truth.csv"``.

    Returns
    -------
    dict of {str: int} or None
        Mapping of normalised participant identifiers (string digits, e.g.
        ``"4"``) to their integer diagnosis labels. Returns ``None`` if the
        file does not exist or cannot be parsed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: Ground truth file not found at {csv_path}.")
        return None

    ground_truth = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            pid, label = parts[0], parts[1]
            if pid.lower() in ("person_id", "id", "person"):
                continue
            try:
                ground_truth[pid] = int(label)
            except ValueError:
                continue

    if not ground_truth:
        print(f"ERROR: No valid records found in {csv_path}.")
        return None

    print(f"Ground truth loaded from: {csv_path} ({len(ground_truth)} participants)")
    return ground_truth


def load_pipeline_scores(json_path=None):
    """
    Loads final behavioural atypicality scores produced by
    ``save_detailed_results()`` in ``disability_engine.py``.

    The function attempts to locate the JSON file at a series of candidate
    paths. Participant identifiers are normalised by stripping the
    ``"Person_"`` prefix (case-insensitive) to ensure consistent matching
    against ground-truth labels.

    Parameters
    ----------
    json_path : str or Path, optional
        Explicit path to the ``behavioral_classification.json`` file.
        If ``None``, the function falls back to a set of default search
        paths relative to the working directory and the script location.

    Returns
    -------
    dict of {str: float} or None
        Mapping of normalised participant identifiers to their final
        behavioural atypicality scores. Returns ``None`` if no valid
        JSON file is found at any of the candidate paths.
    """
    search_paths = [
        json_path,
        "data/output/disability_results/behavioral_classification.json",
        Path(__file__).parent / "data/output/disability_results/behavioral_classification.json",
    ]

    for path in search_paths:
        if path is None:
            return None  
        if not os.path.exists(path):
            return None
        path = Path(path)
        if path.exists():
            print(f"Scores loaded from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = {}
            for pid, info in data.get("persons", {}).items():
                normalized = str(pid).strip()
                if normalized.lower().startswith("person_"):
                    normalized = normalized.split("_", 1)[1]
                scores[normalized] = float(info["final_score"])
            return scores

    print("ERROR: behavioral_classification.json not found.")
    print("Run the main pipeline (main.py) first to generate the scores.")
    return None

def optimize_threshold(scores_dict, ground_truth, n_steps=200):
    """
    Performs a grid search over binary classification thresholds to identify
    the value that maximises the F1-score against the provided ground truth.

    Ties in F1-score are broken by selecting the threshold with the higher
    accuracy. The AUC-ROC is computed once for the best configuration using
    the continuous scores.

    Parameters
    ----------
    scores_dict : dict of {str: float}
        Mapping of normalised participant identifiers to behavioural
        atypicality scores.

    ground_truth : dict of {str: int}
        Mapping of normalised participant identifiers to binary group
        labels (``0`` = healthy, ``1`` = motor-impaired).

    n_steps : int, optional
        Number of evenly spaced threshold candidates between the minimum
        and maximum observed score. Defaults to ``200``.

    Returns
    -------
    dict or None
        Dictionary containing the optimal threshold and associated metrics:

        - ``threshold`` : float — optimal binary classification threshold.
        - ``f1`` : float — F1-score at the optimal threshold.
        - ``accuracy`` : float — accuracy at the optimal threshold.
        - ``auc`` : float — AUC-ROC computed on continuous scores.
        - ``tp``, ``tn``, ``fp``, ``fn`` : int — confusion matrix entries.
        - ``n_positive_predicted`` : int — number of positive predictions.
        - ``n_positive_true`` : int — number of true positive cases.
        - ``persons_evaluated`` : int — number of matched participants.

        Returns ``None`` if no participant identifiers are shared between
        ``scores_dict`` and ``ground_truth``.
    """
    common_ids = [pid for pid in scores_dict if pid in ground_truth]
    if not common_ids:
        print("ERROR: No common participant IDs between scores and ground truth.")
        print(f"IDs in scores: {sorted(scores_dict.keys())}")
        print(f"IDs in ground truth: {sorted(ground_truth.keys())}")
        return None

    missing_from_gt = [pid for pid in scores_dict if pid not in ground_truth]
    missing_from_scores = [pid for pid in ground_truth if pid not in scores_dict]
    if missing_from_gt:
        print(f"Warning: {len(missing_from_gt)} participant(s) in scores have no ground truth: {missing_from_gt}")
    if missing_from_scores:
        print(f"Warning: {len(missing_from_scores)} participant(s) in ground truth have no scores: {missing_from_scores}")

    y_scores = np.array([scores_dict[pid] for pid in common_ids])
    y_true = np.array([ground_truth[pid]  for pid in common_ids])

    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_steps)

    best = {"threshold": None, "f1": -1, "accuracy": -1, "auc": -1}

    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        if f1 > best["f1"] or (f1 == best["f1"] and acc > best["accuracy"]):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception:
                auc = float("nan")
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "accuracy": float(acc),
                "auc": float(auc),
                "tp": int(tp), "tn": int(tn),
                "fp": int(fp), "fn": int(fn),
                "n_positive_predicted": int(np.sum(y_pred)),
                "n_positive_true": int(np.sum(y_true)),
                "persons_evaluated": len(common_ids),
            }

    return best


def optimize_weights(pipeline_fn, ground_truth, weight_steps=5):
    """
    Performs a grid search over the composite score weight vector
    (distance, mahalanobis, consistency) to identify the combination that
    maximises the F1-score.

    Only weight triplets that sum to exactly 1.0 (within a tolerance of
    1e-6) are evaluated. For each candidate triplet, the full pipeline is
    re-run via ``pipeline_fn`` and the optimal threshold is estimated using
    ``optimize_threshold``.

    Parameters
    ----------
    pipeline_fn : callable or None
        A function accepting ``(w_distance, w_mahalanobis, w_consistency)``
        as positional arguments and returning a ``dict`` mapping normalised
        participant identifiers to their final scores. If ``None``, the
        optimisation step is skipped.

    ground_truth : dict of {str: int}
        Mapping of normalised participant identifiers to binary group labels.

    weight_steps : int, optional
        Number of evenly spaced values per weight dimension, producing a
        grid of step size ``1 / weight_steps``. Defaults to ``5``.

    Returns
    -------
    dict or None
        Dictionary containing the optimal weight configuration and
        associated metrics:

        - ``weights`` : dict — optimal values for ``distance``,
          ``mahalanobis``, and ``consistency``.
        - ``f1`` : float — F1-score achieved at the optimal weights.
        - ``accuracy`` : float — accuracy at the optimal weights.
        - ``threshold`` : float — binary threshold optimal for those weights.
        - ``tp``, ``tn``, ``fp``, ``fn`` : int — confusion matrix entries.

        Returns ``None`` if ``pipeline_fn`` is ``None`` or no valid
        weight combination yields a non-null result from ``optimize_threshold``.

    Notes
    -----
    - The search space grows as O(weight_steps²); at ``weight_steps=5``
      approximately 21 valid triplets are evaluated.
    - Progress is reported to standard output every 50 tested combinations.
    """
    if pipeline_fn is None:
        print("pipeline_fn is not available. Skipping weight optimisation.")
        return None

    step = 1.0 / weight_steps
    candidates = np.arange(0, 1 + step, step)

    best = {"weights": None, "f1": -1, "accuracy": -1, "threshold": None}

    total = sum(
        1 for w1, w2, w3 in itertools.product(candidates, repeat=3)
        if abs(w1 + w2 + w3 - 1.0) < 1e-6
    )

    print(f"Weight grid search: {total} candidate combinations to evaluate...")
    tested = 0

    for w1, w2, w3 in itertools.product(candidates, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) > 1e-6:
            continue

        scores = pipeline_fn(w1, w2, w3)
        result = optimize_threshold(scores, ground_truth, n_steps=100)
        if result is None:
            continue

        tested += 1
        if result["f1"] > best["f1"] or (
            result["f1"] == best["f1"] and result["accuracy"] > best["accuracy"]
        ):
            best = {
                "weights": {
                    "distance": float(round(w1, 4)),
                    "mahalanobis": float(round(w2, 4)),
                    "consistency": float(round(w3, 4)),
                },
                "f1": result["f1"],
                "accuracy": result["accuracy"],
                "threshold": result["threshold"],
                "tp": result["tp"], "tn": result["tn"],
                "fp": result["fp"], "fn": result["fn"],
            }

        if tested % 50 == 0:
            print(f"Evaluated {tested}/{total} combinations. Best F1 = {best['f1']:.3f}")

    return best


def suggest_categorical_thresholds(scores_dict, ground_truth, binary_threshold):
    """
    Derives suggested categorical score thresholds (LOW, MEDIUM, HIGH) from
    the optimal binary threshold and the empirical distribution of scores
    among motor-impaired participants.

    The heuristic positions the thresholds as follows:

    - ``LOW``    ≈ 85 % of ``binary_threshold`` (uncertainty zone boundary).
    - ``MEDIUM`` = ``binary_threshold`` (primary classification boundary).
    - ``HIGH``   = 75th percentile of motor-impaired scores when at least
      four such participants are available; otherwise ``binary_threshold x 1.15``.

    A monotonicity constraint (LOW ≤ MEDIUM ≤ HIGH) is enforced after
    the initial estimates are computed.

    Parameters
    ----------
    scores_dict : dict of {str: float}
        Mapping of normalised participant identifiers to behavioural
        atypicality scores.

    ground_truth : dict of {str: int}
        Mapping of normalised participant identifiers to binary group labels.

    binary_threshold : float
        Optimal binary classification threshold as returned by
        ``optimize_threshold``.

    Returns
    -------
    dict or None
        Dictionary with keys ``"LOW"``, ``"MEDIUM"``, and ``"HIGH"`` mapped
        to rounded float threshold values. Returns ``None`` if no
        motor-impaired participants with available scores are found.
    """
    positive_ids = [pid for pid, label in ground_truth.items() if label == 1 and pid in scores_dict]
    positive_scores = [scores_dict[pid] for pid in positive_ids]

    if not positive_scores:
        return None

    high_thr = float(np.percentile(positive_scores, 75)) if len(positive_scores) >= 4 else binary_threshold * 1.15
    medium_thr = float(binary_threshold)
    low_thr = float(binary_threshold * 0.85)

    # Enforce monotonicity
    low_thr = min(low_thr, medium_thr, high_thr)
    medium_thr = min(max(medium_thr, low_thr), high_thr)
    high_thr = max(high_thr, medium_thr)

    return {
        "LOW": round(low_thr, 4),
        "MEDIUM": round(medium_thr, 4),
        "HIGH": round(high_thr, 4),
    }


def update_config(config_path, new_threshold, new_categorical, new_weights=None):
    """
    Overwrites ``DISABILITY_THRESHOLD``, ``DISABILITY_SCORE_THRESHOLDS``,
    and optionally ``DISABILITY_WEIGHTS`` in ``disability_config.py`` with
    the provided optimised values.

    The function reads the existing source file line by line, replaces the
    relevant variable assignments and dict literal blocks in place, and
    writes the result back to disk. Block replacement is triggered by the
    start of the assignment and terminated by the closing ``}`` of the dict.

    Parameters
    ----------
    config_path : str or Path
        Path to the ``disability_config.py`` file to be updated.

    new_threshold : float
        Optimal binary classification threshold to write into
        ``DISABILITY_THRESHOLD``.

    new_categorical : dict of {str: float}
        Mapping with keys ``"LOW"``, ``"MEDIUM"``, and ``"HIGH"`` containing
        the suggested categorical thresholds for ``DISABILITY_SCORE_THRESHOLDS``.

    new_weights : dict of {str: float}, optional
        Mapping with keys ``"distance"``, ``"mahalanobis"``, and
        ``"consistency"`` for ``DISABILITY_WEIGHTS``. If ``None``, the
        weights block is left unchanged.

    Returns
    -------
    bool
        ``True`` if the file was successfully updated; ``False`` if
        ``config_path`` does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}.")
        return False

    source = config_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    updated = []

    skip_block = False
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("DISABILITY_THRESHOLD"):
            updated.append(f"DISABILITY_THRESHOLD = {new_threshold}")
            skip_block = False
            continue

        if stripped.startswith("DISABILITY_SCORE_THRESHOLDS"):
            updated.append("DISABILITY_SCORE_THRESHOLDS = {")
            updated.append(f"'HIGH': {new_categorical['HIGH']},")
            updated.append(f"'MEDIUM': {new_categorical['MEDIUM']},")
            updated.append(f"'LOW': {new_categorical['LOW']},")
            updated.append("}")
            skip_block = True
            continue

        if new_weights and stripped.startswith("DISABILITY_WEIGHTS"):
            updated.append("DISABILITY_WEIGHTS = {")
            updated.append(f"'distance': {new_weights['distance']},")
            updated.append(f"'mahalanobis': {new_weights['mahalanobis']},")
            updated.append(f"'consistency': {new_weights['consistency']},")
            updated.append("}")
            skip_block = True
            continue

        if skip_block:
            if stripped == "}":
                skip_block = False
            continue

        updated.append(line)

    config_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"disability_config.py updated: {config_path}")
    return True


def main():
    """
    Entry point for the threshold and weight optimisation pipeline.

    Executes the following steps in sequence:

    1. Loads ground-truth labels from the CSV file.
    2. Loads behavioural atypicality scores from the pipeline JSON output.
    3. Prints a per-participant score and label overview.
    4. Runs binary threshold optimisation via grid search.
    5. Derives suggested categorical thresholds (LOW / MEDIUM / HIGH).
    6. Serialises all results to ``optimization_results.json``.
    7. Prompts the user to optionally apply the optimal values to
       ``disability_config.py``.

    Returns
    -------
    None
    """
    print("=" * 65)
    print("Threshold Optimisation Against Ground Truth")
    print("=" * 65)

    ground_truth = load_ground_truth()
    if ground_truth is None:
        return

    scores = load_pipeline_scores()
    if scores is None:
        return

    print(f"\nScores loaded for {len(scores)} participants:")
    for pid, sc in sorted(scores.items(), key=lambda x: x[0]):
        gt_label = ground_truth.get(pid, "?")
        print(f"Person {pid:>3}: score = {sc:.4f}, ground_truth = {gt_label}")

    print("\nBinary Threshold Optimisation")
    result = optimize_threshold(scores, ground_truth)
    if result is None:
        return

    print(f"\nBest threshold found : {result['threshold']:.4f}")
    print(f"F1-score: {result['f1']:.3f}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"AUC-ROC: {result['auc']:.3f}")
    print(f"TP={result['tp']}  TN={result['tn']}  FP={result['fp']}  FN={result['fn']}")
    print(f"Participants evaluated: {result['persons_evaluated']}")

    categorical = suggest_categorical_thresholds(scores, ground_truth, result["threshold"])
    if categorical:
        print(f"\nSuggested categorical thresholds:")
        print(f"LOW: {categorical['LOW']}")
        print(f"MEDIUM: {categorical['MEDIUM']}")
        print(f"HIGH: {categorical['HIGH']}")

    output_dir = Path("data/output/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "optimization_results.json"

    output_data = {
        "binary_threshold_optimization": result,
        "categorical_thresholds_suggested": categorical,
        "ground_truth_used": ground_truth,
        "scores_used": scores,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()