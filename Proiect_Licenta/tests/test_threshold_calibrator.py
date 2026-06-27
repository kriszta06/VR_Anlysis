import csv
import json
import os
import tempfile

import numpy as np
import pytest

from src.evaluation import threshold_calibrator


def write_json(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def write_csv(rows, fieldnames) -> str:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()
    return tmp.name


class TestLoadScoresFromResults:

    def test_loads_final_scores_by_person(self):
        path = write_json({
            "persons": {
                "Person_1": {"final_score": 0.42},
                "Person_2": {"final_score": 0.87},
            }
        })
        try:
            scores = threshold_calibrator.load_scores_from_results(path)
            assert scores == {"Person_1": 0.42, "Person_2": 0.87}
        finally:
            os.unlink(path)

    def test_empty_persons_returns_empty_dict(self):
        path = write_json({"persons": {}})
        try:
            scores = threshold_calibrator.load_scores_from_results(path)
            assert scores == {}
        finally:
            os.unlink(path)


class TestLoadGroundTruthCsv:

    def test_standard_headers(self):
        path = write_csv(
            [{"person_id": "1", "label": "1"}, {"person_id": "2", "label": "0"}],
            fieldnames=["person_id", "label"],
        )
        try:
            gt = threshold_calibrator.load_ground_truth_csv(path)
            assert gt == {"Person_1": 1, "Person_2": 0}
        finally:
            os.unlink(path)

    def test_alternate_headers_id_and_diagnosis(self):
        path = write_csv(
            [{"id": "5", "diagnosis": "1"}],
            fieldnames=["id", "diagnosis"],
        )
        try:
            gt = threshold_calibrator.load_ground_truth_csv(path)
            assert gt == {"Person_5": 1}
        finally:
            os.unlink(path)

    def test_invalid_label_value_is_skipped(self):
        path = write_csv(
            [{"person_id": "1", "label": "not_a_number"}, {"person_id": "2", "label": "1"}],
            fieldnames=["person_id", "label"],
        )
        try:
            gt = threshold_calibrator.load_ground_truth_csv(path)
            assert gt == {"Person_2": 1}
        finally:
            os.unlink(path)

    def test_missing_required_columns_returns_empty_dict(self):
        path = write_csv(
            [{"name": "1", "value": "1"}],
            fieldnames=["name", "value"],
        )
        try:
            gt = threshold_calibrator.load_ground_truth_csv(path)
            assert gt == {}
        finally:
            os.unlink(path)

    def test_whitespace_in_person_id_is_trimmed(self):
        path = write_csv(
            [{"person_id": "  3  ", "label": "0"}],
            fieldnames=["person_id", "label"],
        )
        try:
            gt = threshold_calibrator.load_ground_truth_csv(path)
            assert gt == {"Person_3": 0}
        finally:
            os.unlink(path)


class TestCalibrateThreshold:

    def test_returns_threshold_and_auc(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        threshold, roc_auc = threshold_calibrator.calibrate_threshold(y_true, y_score)
        assert isinstance(threshold, float)
        assert isinstance(roc_auc, float)
        assert 0.0 <= roc_auc <= 1.0

    def test_perfect_separation_gives_auc_one(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        _, roc_auc = threshold_calibrator.calibrate_threshold(y_true, y_score)
        assert roc_auc == pytest.approx(1.0)

    def test_threshold_within_score_range(self):
        y_true = [0, 0, 1, 1, 1]
        y_score = [0.2, 0.3, 0.5, 0.6, 0.95]
        threshold, _ = threshold_calibrator.calibrate_threshold(y_true, y_score)
        assert min(y_score) <= threshold <= max(y_score) + 1


class TestAnalyzePerson:

    def test_person_not_found_returns_none(self):
        scores = {"Person_1": 0.6}
        ground_truth = {"Person_1": 1}
        result = threshold_calibrator.analyze_person("Person_99", scores, ground_truth)
        assert result is None

    def test_correct_prediction_above_threshold(self):
        scores = {"Person_1": 0.7}
        ground_truth = {"Person_1": 1}
        result = threshold_calibrator.analyze_person("Person_1", scores, ground_truth)
        assert result["prediction_at_0_5"] == 1
        assert result["correct_at_0_5"] is True

    def test_prediction_below_threshold(self):
        scores = {"Person_1": 0.3}
        ground_truth = {"Person_1": 1}
        result = threshold_calibrator.analyze_person("Person_1", scores, ground_truth)
        assert result["prediction_at_0_5"] == 0
        assert result["correct_at_0_5"] is False

    def test_missing_ground_truth_gives_none_correctness(self):
        scores = {"Person_1": 0.8}
        ground_truth = {}
        result = threshold_calibrator.analyze_person("Person_1", scores, ground_truth)
        assert result["ground_truth"] is None
        assert result["correct_at_0_5"] is None


class TestMain:

    def test_missing_results_path_prints_error(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(threshold_calibrator.config, "RESULTS_PATH", str(tmp_path / "missing.json"), raising=False)
        threshold_calibrator.main()
        captured = capsys.readouterr()
        assert "Classification results file not found" in captured.out

    def test_missing_ground_truth_path_prints_error(self, monkeypatch, tmp_path, capsys):
        results_path = write_json({"persons": {"Person_1": {"final_score": 0.5}}})
        monkeypatch.setattr(threshold_calibrator.config, "RESULTS_PATH", results_path, raising=False)
        monkeypatch.setattr(threshold_calibrator.config, "GROUND_TRUTH_PATH", str(tmp_path / "missing.csv"), raising=False)
        try:
            threshold_calibrator.main()
            captured = capsys.readouterr()
            assert "Ground truth file not found" in captured.out
        finally:
            os.unlink(results_path)

    def test_no_common_persons_prints_error(self, monkeypatch, capsys):
        results_path = write_json({"persons": {"Person_1": {"final_score": 0.5}}})
        gt_path = write_csv([{"person_id": "99", "label": "1"}], fieldnames=["person_id", "label"])
        monkeypatch.setattr(threshold_calibrator.config, "RESULTS_PATH", results_path, raising=False)
        monkeypatch.setattr(threshold_calibrator.config, "GROUND_TRUTH_PATH", gt_path, raising=False)
        try:
            threshold_calibrator.main()
            captured = capsys.readouterr()
            assert "No common persons found" in captured.out
        finally:
            os.unlink(results_path)
            os.unlink(gt_path)

    def test_successful_run_prints_calibration_results(self, monkeypatch, capsys):
        persons = {
            f"Person_{i}": {"final_score": 0.1 * i}
            for i in range(1, 6)
        }
        results_path = write_json({"persons": persons})
        gt_rows = [{"person_id": str(i), "label": str(int(i >= 4))} for i in range(1, 6)]
        gt_path = write_csv(gt_rows, fieldnames=["person_id", "label"])
        monkeypatch.setattr(threshold_calibrator.config, "RESULTS_PATH", results_path, raising=False)
        monkeypatch.setattr(threshold_calibrator.config, "GROUND_TRUTH_PATH", gt_path, raising=False)
        try:
            threshold_calibrator.main()
            captured = capsys.readouterr()
            assert "ROC THRESHOLD CALIBRATION" in captured.out
            assert "Optimal Threshold" in captured.out
            assert "PERSON ANALYSIS" in captured.out
        finally:
            os.unlink(results_path)
            os.unlink(gt_path)