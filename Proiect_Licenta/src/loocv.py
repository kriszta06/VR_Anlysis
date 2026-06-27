import sys
import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "output"

from src.core.analysis.disability_engine import analyze_person_disability
from src.evaluation.ground_truth_handler import load_ground_truth
from src.config import disability_config as config

def find_optimal_threshold_youden(scores_train, labels_train):
    """
    Determines the optimal binary classification threshold using Youden's
    J statistic on the training data.

    The optimal threshold is selected by maximizing Youden's J statistic
    (``TPR - FPR``), which identifies the operating point that provides the
    best trade-off between sensitivity and specificity. The function also
    computes the Area Under the ROC Curve (AUC) for the training fold.

    If the training labels contain only a single class, the ROC curve cannot
    be computed. In this case, the function returns the default threshold
    defined in the project configuration together with ``NaN`` for the AUC.

    Parameters
    ----------
    scores_train : array-like
        Predicted scores or probabilities for the training samples.

    labels_train : array-like
        Ground-truth binary labels corresponding to the training samples.

    Returns
    -------
    tuple[float, float]
        A tuple containing:

        - ``optimal_threshold`` : float
            Threshold that maximizes Youden's J statistic.
        - ``auc`` : float
            Area Under the ROC Curve (AUC) computed on the training data.
            Returns ``NaN`` if the training fold contains only one class.

    Notes
    -----
    - The threshold optimization is performed exclusively on the training fold
    to avoid information leakage.
    - When only one class is present in the training data, the fallback
    threshold specified by ``config.DISABILITY_THRESHOLD`` is returned.
    """
    if len(set(labels_train)) < 2:
        fallback = getattr(config, 'DISABILITY_THRESHOLD', 0.55)
        return fallback, float('nan')

    fpr, tpr, thresholds = roc_curve(labels_train, scores_train)
    auc = roc_auc_score(labels_train, scores_train)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx]), float(auc)


def find_optimal_threshold_f1(scores_train, labels_train, n_steps=200):
    """
    Determines the optimal binary classification threshold by maximizing the
    F1-score on the training data.

    The function evaluates a series of candidate thresholds uniformly sampled
    between the minimum and maximum predicted scores. For each threshold, binary
    predictions are generated and the corresponding F1-score is computed. The
    threshold yielding the highest F1-score is returned.

    This method serves as an alternative threshold optimization strategy and can
    be used when threshold selection based on Youden's J statistic produces
    unsatisfactory results.

    Parameters
    ----------
    scores_train : array-like
        Predicted scores or probabilities for the training samples.

    labels_train : array-like
        Ground-truth binary labels corresponding to the training samples.

    n_steps : int, default=200
        Number of candidate thresholds evaluated between the minimum and maximum
        predicted scores.

    Returns
    -------
    float
        Threshold that maximizes the F1-score on the training data.

    Notes
    -----
    - Threshold optimization is performed exclusively on the training fold to
    prevent information leakage.
    - If the training data contain only one class, the default threshold
    specified by ``config.DISABILITY_THRESHOLD`` is returned.
    - The F1-score is computed with ``zero_division=0`` to avoid undefined
    values when no positive predictions are produced.
    """
    if len(set(labels_train)) < 2:
        return getattr(config, 'DISABILITY_THRESHOLD', 0.55)

    thresholds = np.linspace(min(scores_train), max(scores_train), n_steps)
    best_thr, best_f1 = thresholds[0], -1.0

    for thr in thresholds:
        preds = (np.array(scores_train) >= thr).astype(int)
        f1 = f1_score(labels_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_thr)


def run_loocv(person_aggregated_data, ground_truth, threshold_method="youden"):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) for the behavioral
    variation detection pipeline using participant-level aggregated feature
    vectors.

    For each cross-validation fold, one participant is held out as the test
    sample while the remaining participants are used for training. The
    behavioral variation analysis is performed on the training subset, an
    optimal decision threshold is estimated using the selected threshold
    optimization method, and the held-out participant is subsequently evaluated.
    The function aggregates the predictions from all folds and computes the
    overall classification performance.

    Parameters
    ----------
    person_aggregated_data : dict[str, numpy.ndarray]
        Dictionary mapping participant identifiers to their aggregated feature
        vectors.

    ground_truth : dict[str, int]
        Dictionary mapping participant identifiers to their binary ground-truth
        labels, where 1 denotes behavioral variation and 0 denotes typical
        behavior.

    threshold_method : {"youden", "f1"}, default="youden"
        Method used to determine the decision threshold on each training fold.

        - ``"youden"``: maximizes Youden's J statistic.
        - ``"f1"``: maximizes the F1-score.

    Returns
    -------
    dict or None
        Dictionary containing the complete LOOCV evaluation results, including:

        - ``method`` : str
            Cross-validation protocol.
        - ``threshold_method`` : str
            Threshold optimization method.
        - ``n_persons`` : int
            Number of evaluated participants.
        - ``n_disabled`` : int
            Number of participants labeled with behavioral variation.
        - ``n_healthy`` : int
            Number of participants labeled as typical.
        - ``metrics`` : dict
            Overall classification metrics, including accuracy, F1-score,
            weighted F1-score, ROC-AUC, sensitivity, specificity, and positive
            predictive value (PPV).
        - ``confusion_matrix`` : dict
            Numbers of true positives, true negatives, false positives, and
            false negatives.
        - ``threshold_stats`` : dict
            Summary statistics of the thresholds estimated across all folds.
        - ``per_person`` : list of dict
            Prediction results for each evaluated participant.

        Returns ``None`` if fewer than four participants with ground-truth
        labels are available.

    Notes
    -----
    - Participants without corresponding ground-truth labels are excluded from
    the evaluation.
    - Decision thresholds are calibrated exclusively on the training fold to
    prevent information leakage.
    - The held-out participant is scored in the context of the complete group,
    while the decision threshold remains derived solely from the training
    participants.
    - Overall performance metrics are computed from the out-of-sample
    predictions collected across all LOOCV folds.
    - Progress information and fold-specific results are printed during the
    evaluation.
    """
    all_person_ids = list(person_aggregated_data.keys())
    n = len(all_person_ids)

    print(f"\nNumber of LOOCV participants: {n}, thresholding method: {threshold_method.upper()}\n")

    persons_with_gt = [p for p in all_person_ids if p in ground_truth]
    missing_gt = [p for p in all_person_ids if p not in ground_truth]
    if missing_gt:
        print(f"Error: Missing ground truth labels for participants: {', '.join(missing_gt)}")

    if len(persons_with_gt) < 4:
        print("Error: Not enough participants with ground truth labels for LOOCV evaluation.")
        return None

    loocv_records = []       
    per_fold_thresholds = [] 

    for i, test_person in enumerate(persons_with_gt):

        train_persons = [p for p in persons_with_gt if p != test_person]
        train_data = {p: person_aggregated_data[p] for p in train_persons}

        try:
            train_assessment = analyze_person_disability(train_data)
        except Exception as e:
            print(f"  [Fold {i+1}/{n}] Error for participant {test_person}: {e} (test fold)")
            continue

        train_scores  = [train_assessment[p]['final_score'] for p in train_persons]
        train_labels  = [int(ground_truth[p]) for p in train_persons]

        if threshold_method == "youden":
            threshold, train_auc = find_optimal_threshold_youden(train_scores, train_labels)
        else:
            threshold = find_optimal_threshold_f1(train_scores, train_labels)
            train_auc = float('nan')

        per_fold_thresholds.append(threshold)

        full_data_for_scoring = {**train_data, test_person: person_aggregated_data[test_person]}

        try:
            full_assessment = analyze_person_disability(full_data_for_scoring)
        except Exception as e:
            print(f"  [Fold {i+1}/{n}] Error for participant {test_person}: {e} (scoring fold)")
            continue

        test_score      = full_assessment[test_person]['final_score']
        test_true_label = int(ground_truth[test_person])
        test_pred_label = int(test_score >= threshold)
        correct         = test_pred_label == test_true_label
        status          = "OK" if correct else "WRONG"

        print(
            f"Fold {i+1:>2}/{n} | {test_person:<15} | "
            f"GT={test_true_label} | Score={test_score:.4f} | "
            f"Thr={threshold:.4f} | Pred={test_pred_label} | {status}"
        )

        loocv_records.append({
            "person": test_person,
            "true_label": test_true_label,
            "final_score": round(float(test_score), 6),
            "threshold_used": round(float(threshold), 6),
            "predicted_label": test_pred_label,
            "correct": correct,
            "train_auc": round(float(train_auc), 4) if not np.isnan(train_auc) else None,
        })

    y_true   = [r["true_label"] for r in loocv_records]
    y_pred   = [r["predicted_label"] for r in loocv_records]
    y_scores = [r["final_score"] for r in loocv_records]

    accuracy  = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan') 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')  
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')  

    mean_thr = float(np.mean(per_fold_thresholds))
    std_thr  = float(np.std(per_fold_thresholds))

    results = {
        "method": "Leave-One-Out Cross-Validation (LOOCV)",
        "threshold_method": threshold_method,
        "n_persons": len(loocv_records),
        "n_disabled": int(sum(y_true)),
        "n_healthy": int(len(y_true) - sum(y_true)),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "f1_weighted": round(f1_weighted, 4),
            "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
            "sensitivity": round(sensitivity, 4) if not np.isnan(sensitivity) else None,
            "specificity": round(specificity, 4) if not np.isnan(specificity) else None,
            "ppv_precision": round(ppv, 4) if not np.isnan(ppv) else None,
        },
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
        },
        "threshold_stats": {
            "mean": round(mean_thr, 4),
            "std": round(std_thr,  4),
            "min": round(float(min(per_fold_thresholds)), 4),
            "max": round(float(max(per_fold_thresholds)), 4),
            "all_folds": [round(t, 4) for t in per_fold_thresholds],
        },
        "per_person": loocv_records,
    }

    return results

def print_loocv_summary(results):
    """
    Prints a formatted summary of the Leave-One-Out Cross-Validation (LOOCV)
    evaluation results.

    The function displays the overall classification performance, confusion
    matrix, threshold calibration statistics, and a qualitative interpretation
    of the obtained ROC-AUC value. The summary is intended for console output
    only and provides a concise overview of the model's out-of-sample
    performance.

    Parameters
    ----------
    results : dict
        Dictionary containing the LOOCV evaluation results produced by
        ``run_loocv()``. The dictionary is expected to include the following
        keys:

        - ``metrics``
        - ``confusion_matrix``
        - ``threshold_stats``
        - ``n_persons``
        - ``n_disabled``
        - ``n_healthy``

    Returns
    -------
    None
        The function does not return a value. It prints a formatted summary of
        the evaluation results to the console.

    Notes
    -----
    - The reported metrics include ROC-AUC, F1-score, weighted F1-score,
    accuracy, sensitivity, specificity, and positive predictive value (PPV).
    - The confusion matrix is presented as the numbers of true positives, true
    negatives, false positives, and false negatives.
    - Threshold statistics summarize the decision thresholds estimated across
    all LOOCV folds.
    - A qualitative interpretation of the ROC-AUC value is printed to provide
    an overall assessment of the model's generalization performance.
    """
    m   = results["metrics"]
    cm  = results["confusion_matrix"]
    thr = results["threshold_stats"]

    print(f"\n{'='*60}")
    print("  RESULTS FOR LOOCV — OUT-OF-SAMPLE")
    print(f"{'='*60}")
    print(f" Number of participants: {results['n_persons']} "
          f"({results['n_disabled']} with atypical movement, {results['n_healthy']} with typical movement)")
    print(f"\n  Metrics:")
    print(f"AUC-ROC: {m['auc_roc']}")
    print(f"F1-score: {m['f1_score']}")
    print(f"F1 weighted: {m['f1_weighted']}")
    print(f"Accuracy: {m['accuracy']:.1%}")
    print(f"Sensitivitate: {m['sensitivity']} (detectia dizabilitatii)")
    print(f"Specificitate: {m['specificity']} (excluderea sanatosilor)")
    print(f"PPV/Precision: {m['ppv_precision']}")
    print(f"\n  Confusion Matrix:")
    print(f"TP={cm['TP']}  FN={cm['FN']}")
    print(f"FP={cm['FP']}  TN={cm['TN']}")
    print(f"\nThreshold (calibrated per fold):")
    print(f"Mean: {thr['mean']}  ±  {thr['std']}")
    print(f"Range: [{thr['min']}, {thr['max']}]")
    print(f"{'='*60}")

    auc = m.get('auc_roc')
    f1  = m.get('f1_score')
    if auc is not None:
        if auc >= 0.80:
            verdict = "GOOD - the system has high predictive accuracy."
        elif auc >= 0.65:
            verdict = "MODERATE - the system has moderate predictive accuracy."
        else:
            verdict = "POOR - the system has low predictive accuracy."
        print(f"\n  Interpretation: AUC={auc:.3f} => {verdict}")

    print()


def save_loocv_results(results, output_dir=None):
    """
    Saves the Leave-One-Out Cross-Validation (LOOCV) evaluation results to a
    JSON file.

    The function creates the output directory if it does not already exist,
    serializes the complete evaluation results into a human-readable JSON file,
    and returns the path to the generated file.

    Parameters
    ----------
    results : dict
        Dictionary containing the LOOCV evaluation results, typically produced
        by ``run_loocv()``.

    output_dir : str or pathlib.Path, optional
        Directory in which the JSON file will be stored. Defaults to
        ``src/data/output/loocv``. The directory is created automatically if it
        does not already exist.

    Returns
    -------
    pathlib.Path
        Path to the generated ``loocv_results.json`` file.

    Outputs
    -------
    The following file is generated:

    - ``loocv_results.json``

    stored in the specified output directory.

    Notes
    -----
    - The results are saved using UTF-8 encoding with indentation to improve
    readability.
    - Existing files with the same name are overwritten.
    - The path to the saved file is printed to the console.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "loocv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "loocv_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {out_path}")
    return out_path

def run_loocv_from_json(
    json_path="data/output/disability_results/behavioral_classification.json",
    ground_truth_path="data/ground_truth/ground_truth.csv",
    threshold_method="youden",
):
    """
    Loads participant feature vectors from a previously generated JSON file and
    performs Leave-One-Out Cross-Validation (LOOCV) using the available
    ground-truth labels.

    The function reads the aggregated participant feature vectors produced by
    the behavioral analysis pipeline, loads the corresponding ground-truth
    labels, normalizes participant identifiers to ensure consistency between
    both datasets, executes the LOOCV evaluation, prints a summary of the
    results, and saves the complete evaluation to a JSON file.

    Parameters
    ----------
    json_path : str or pathlib.Path, \
            default="data/output/disability_results/behavioral_classification.json"
        Path to the JSON file containing the aggregated participant feature
        vectors.

    ground_truth_path : str or pathlib.Path, \
            default="data/ground_truth/ground_truth.csv"
        Path to the CSV file containing the binary ground-truth labels.

    threshold_method : {"youden", "f1"}, default="youden"
        Threshold optimization strategy used during LOOCV.

        - ``"youden"``: maximizes Youden's J statistic.
        - ``"f1"``: maximizes the F1-score.

    Returns
    -------
    dict or None
        Dictionary containing the complete LOOCV evaluation results returned by
        ``run_loocv()``. Returns ``None`` if the required input files cannot be
        loaded or no valid participant feature vectors are available.

    Notes
    -----
    - Participant identifiers are normalized by removing the ``"Person_"``
    prefix, when present, to ensure consistency between the feature vectors
    and the ground-truth labels.
    - The function requires the behavioral classification JSON file generated
    by the main analysis pipeline.
    - Upon successful evaluation, a summary of the LOOCV results is printed to
    the console and the complete results are saved as a JSON file using
    ``save_loocv_results()``.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Error: Classification results file not found: {json_path}. Run the main analysis pipeline first.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    person_aggregated_data = {}
    for pid, info in data.get("persons", {}).items():
        normalized = str(pid).strip()
        if normalized.lower().startswith("person_"):
            normalized = normalized.split("_", 1)[1]
        raw = info.get("raw_features")
        if raw is not None:
            person_aggregated_data[normalized] = np.array(raw)

    if not person_aggregated_data:
        print("Error: No valid participant feature vectors found.")
        return None

    ground_truth = load_ground_truth(ground_truth_path)
    if not ground_truth:
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        return None

    gt_normalized = {}
    for k, v in ground_truth.items():
        key = str(k).strip()
        if key.lower().startswith("person_"):
            key = key.split("_", 1)[1]
        gt_normalized[key] = int(v)

    results = run_loocv(person_aggregated_data, gt_normalized, threshold_method)
    if results:
        print_loocv_summary(results)
        save_loocv_results(results)
    return results


if __name__ == "__main__":
    """
    Example usage:
        python loocv.py # Runs Leave-One-Out Cross-Validation using Youden's J statistic.
        python loocv.py f1 # Runs Leave-One-Out Cross-Validation using F1-score threshold optimization.
    """
    method = sys.argv[1] if len(sys.argv) > 1 else "youden"
    run_loocv_from_json(threshold_method=method)