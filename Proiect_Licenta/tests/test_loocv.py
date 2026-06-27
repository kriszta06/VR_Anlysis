import json
import os
import tempfile

import numpy as np
import pytest

from src import loocv



def write_json(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def fake_analyze_person_disability(person_data):
    return {p: {'final_score': float(np.asarray(v).ravel()[0])} for p, v in person_data.items()}


def make_sample_results():
    return {
        "n_persons": 5,
        "n_disabled": 2,
        "n_healthy": 3,
        "metrics": {
            "auc_roc": 0.85,
            "f1_score": 0.75,
            "f1_weighted": 0.78,
            "accuracy": 0.8,
            "sensitivity": 0.7,
            "specificity": 0.9,
            "ppv_precision": 0.8,
        },
        "confusion_matrix": {"TP": 2, "TN": 2, "FP": 1, "FN": 0},
        "threshold_stats": {"mean": 0.55, "std": 0.05, "min": 0.5, "max": 0.6},
    }


class TestFindOptimalThresholdYouden:

    def test_single_class_returns_fallback_and_nan(self, monkeypatch):
        monkeypatch.setattr(loocv.config, "DISABILITY_THRESHOLD", 0.6, raising=False)
        threshold, auc = loocv.find_optimal_threshold_youden([0.1, 0.2, 0.3], [0, 0, 0])
        assert threshold == 0.6
        assert np.isnan(auc)

    def test_two_classes_returns_float_threshold_and_auc(self):
        scores = [0.1, 0.2, 0.6, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        threshold, auc = loocv.find_optimal_threshold_youden(scores, labels)
        assert isinstance(threshold, float)
        assert isinstance(auc, float)
        assert not np.isnan(auc)

    def test_perfect_separation_gives_auc_one(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        _, auc = loocv.find_optimal_threshold_youden(scores, labels)
        assert auc == pytest.approx(1.0)


class TestFindOptimalThresholdF1:

    def test_single_class_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(loocv.config, "DISABILITY_THRESHOLD", 0.6, raising=False)
        threshold = loocv.find_optimal_threshold_f1([0.1, 0.2, 0.3], [1, 1, 1])
        assert threshold == 0.6

    def test_two_classes_returns_float_threshold(self):
        scores = [0.1, 0.2, 0.6, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        threshold = loocv.find_optimal_threshold_f1(scores, labels)
        assert isinstance(threshold, float)
        assert min(scores) <= threshold <= max(scores)

    def test_custom_n_steps(self):
        scores = [0.0, 0.5, 1.0]
        labels = [0, 1, 1]
        threshold = loocv.find_optimal_threshold_f1(scores, labels, n_steps=10)
        assert isinstance(threshold, float)


class TestRunLoocv:

    def test_returns_none_when_fewer_than_four_with_ground_truth(self, monkeypatch):
        monkeypatch.setattr(loocv, "analyze_person_disability", fake_analyze_person_disability)
        person_data = {f"P{i}": np.array([0.1 * i]) for i in range(3)}
        ground_truth = {f"P{i}": i % 2 for i in range(3)}
        result = loocv.run_loocv(person_data, ground_truth)
        assert result is None

    def test_basic_execution_youden(self, monkeypatch):
        monkeypatch.setattr(loocv, "analyze_person_disability", fake_analyze_person_disability)
        person_data = {f"P{i}": np.array([0.1 * i]) for i in range(5)}
        ground_truth = {f"P{i}": int(i >= 3) for i in range(5)}
        result = loocv.run_loocv(person_data, ground_truth, threshold_method="youden")
        assert result is not None
        assert result["n_persons"] == 5
        assert "metrics" in result
        assert "confusion_matrix" in result
        assert "threshold_stats" in result
        assert len(result["per_person"]) == 5

    def test_basic_execution_f1(self, monkeypatch):
        monkeypatch.setattr(loocv, "analyze_person_disability", fake_analyze_person_disability)
        person_data = {f"P{i}": np.array([0.1 * i]) for i in range(5)}
        ground_truth = {f"P{i}": int(i >= 3) for i in range(5)}
        result = loocv.run_loocv(person_data, ground_truth, threshold_method="f1")
        assert result is not None
        assert result["threshold_method"] == "f1"

    def test_missing_ground_truth_persons_are_excluded(self, monkeypatch, capsys):
        monkeypatch.setattr(loocv, "analyze_person_disability", fake_analyze_person_disability)
        person_data = {f"P{i}": np.array([0.1 * i]) for i in range(6)}
        ground_truth = {f"P{i}": int(i >= 3) for i in range(5)}
        result = loocv.run_loocv(person_data, ground_truth)
        captured = capsys.readouterr()
        assert "Missing ground truth labels" in captured.out
        assert result["n_persons"] == 5

    def test_analyze_exception_skips_fold(self, monkeypatch, capsys):
        call_counter = {"count": 0}

        def flaky_analyze(data):
            call_counter["count"] += 1
            if call_counter["count"] == 2:
                raise ValueError("boom")
            return fake_analyze_person_disability(data)

        monkeypatch.setattr(loocv, "analyze_person_disability", flaky_analyze)
        person_data = {f"P{i}": np.array([0.1 * i]) for i in range(5)}
        ground_truth = {f"P{i}": int(i >= 3) for i in range(5)}
        result = loocv.run_loocv(person_data, ground_truth)
        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert result["n_persons"] == 4


class TestPrintLoocvSummary:

    def test_prints_summary_without_error(self, capsys):
        results = make_sample_results()
        loocv.print_loocv_summary(results)
        captured = capsys.readouterr()
        assert "RESULTS FOR LOOCV" in captured.out
        assert "AUC-ROC" in captured.out

    def test_prints_good_interpretation_for_high_auc(self, capsys):
        results = make_sample_results()
        results["metrics"]["auc_roc"] = 0.9
        loocv.print_loocv_summary(results)
        captured = capsys.readouterr()
        assert "GOOD" in captured.out

    def test_prints_poor_interpretation_for_low_auc(self, capsys):
        results = make_sample_results()
        results["metrics"]["auc_roc"] = 0.4
        loocv.print_loocv_summary(results)
        captured = capsys.readouterr()
        assert "POOR" in captured.out


class TestSaveLoocvResults:

    def test_creates_json_file(self, tmp_path):
        results = make_sample_results()
        out_path = loocv.save_loocv_results(results, output_dir=str(tmp_path))
        assert out_path.exists()

    def test_saved_content_matches_input(self, tmp_path):
        results = make_sample_results()
        out_path = loocv.save_loocv_results(results, output_dir=str(tmp_path))
        with open(out_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == results

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested_dir = tmp_path / "nested" / "results"
        results = make_sample_results()
        out_path = loocv.save_loocv_results(results, output_dir=str(nested_dir))
        assert nested_dir.exists()
        assert out_path.exists()


class TestRunLoocvFromJson:

    def test_missing_classification_file_returns_none(self):
        result = loocv.run_loocv_from_json(
            json_path="/nonexistent/classification.json",
            ground_truth_path="/nonexistent/ground_truth.csv",
        )
        assert result is None

    def test_no_valid_feature_vectors_returns_none(self):
        path = write_json({"persons": {"Person_1": {}}})
        try:
            result = loocv.run_loocv_from_json(json_path=path, ground_truth_path="/nonexistent/gt.csv")
            assert result is None
        finally:
            os.unlink(path)

    def test_missing_ground_truth_returns_none(self, monkeypatch):
        monkeypatch.setattr(loocv, "load_ground_truth", lambda path: {})
        path = write_json({"persons": {"Person_1": {"raw_features": [0.1, 0.2]}}})
        try:
            result = loocv.run_loocv_from_json(json_path=path, ground_truth_path="ignored")
            assert result is None
        finally:
            os.unlink(path)

    def test_successful_run_normalizes_person_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(loocv, "analyze_person_disability", fake_analyze_person_disability)
        monkeypatch.setattr(
            loocv, "load_ground_truth",
            lambda path: {str(i): int(i >= 3) for i in range(5)},
        )
        monkeypatch.setattr(loocv, "save_loocv_results", lambda results, output_dir="x": None)

        persons = {f"Person_{i}": {"raw_features": [0.1 * i]} for i in range(5)}
        path = write_json({"persons": persons})
        try:
            result = loocv.run_loocv_from_json(json_path=path, ground_truth_path="ignored")
            assert result is not None
            person_ids = {r["person"] for r in result["per_person"]}
            assert all(not pid.lower().startswith("person_") for pid in person_ids)
        finally:
            os.unlink(path)