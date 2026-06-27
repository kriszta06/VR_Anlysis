import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from src.evaluation import metrics

def write_json(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


class TestGetOutputPath:

    def test_creates_directory_if_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "__file__", str(tmp_path / "nested" / "metrics.py"))
        output_dir = metrics.get_output_path()
        assert os.path.isdir(output_dir)

    def test_returns_expected_subpath(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "__file__", str(tmp_path / "metrics.py"))
        output_dir = metrics.get_output_path()
        assert output_dir.endswith(os.path.join("data", "output", "evaluation_results"))

    def test_idempotent_when_directory_already_exists(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "__file__", str(tmp_path / "metrics.py"))
        first_call = metrics.get_output_path()
        second_call = metrics.get_output_path()
        assert first_call == second_call
        assert os.path.isdir(second_call)


class TestCalculateAndPlotMetrics:

    def test_empty_y_true_prints_error_and_returns_none(self, capsys):
        result = metrics.calculate_and_plot_metrics([], [])
        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert result is None

    def test_basic_binary_classification_saves_plot(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        metrics.calculate_and_plot_metrics(y_true, y_pred)
        plot_path = tmp_path / "confusion_matrix.png"
        assert plot_path.exists()

    def test_explicit_labels_used(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        y_true = ['LOW', 'HIGH', 'MEDIUM']
        y_pred = ['LOW', 'HIGH', 'HIGH']
        metrics.calculate_and_plot_metrics(y_true, y_pred, labels=['NONE', 'LOW', 'MEDIUM', 'HIGH'])
        captured = capsys.readouterr()
        assert "Overall accurcy" in captured.out

    def test_labels_inferred_when_none_provided(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        y_true = [0, 1, 1]
        y_pred = [0, 0, 1]
        metrics.calculate_and_plot_metrics(y_true, y_pred, labels=None)
        plot_path = tmp_path / "confusion_matrix.png"
        assert plot_path.exists()

    def test_perfect_predictions_report_full_accuracy(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        metrics.calculate_and_plot_metrics(y_true, y_pred)
        captured = capsys.readouterr()
        assert "100.00%" in captured.out


class TestExportEvaluationSummary:

    def test_creates_json_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        metrics.export_evaluation_summary([0, 1, 0, 1], [0, 1, 1, 1])
        json_path = tmp_path / "evaluation_summary.json"
        assert json_path.exists()

    def test_summary_contains_expected_keys(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        metrics.export_evaluation_summary([0, 1, 0, 1], [0, 1, 1, 1])
        json_path = tmp_path / "evaluation_summary.json"
        with open(json_path, 'r') as f:
            summary = json.load(f)
        assert "total_samples" in summary
        assert "accuracy_score" in summary
        assert "evaluation_timestamp" in summary
        assert "status" in summary

    def test_total_samples_matches_input_length(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 1, 0]
        metrics.export_evaluation_summary(y_true, y_pred)
        json_path = tmp_path / "evaluation_summary.json"
        with open(json_path, 'r') as f:
            summary = json.load(f)
        assert summary["total_samples"] == len(y_true)

    def test_accuracy_score_is_correct(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]
        metrics.export_evaluation_summary(y_true, y_pred)
        json_path = tmp_path / "evaluation_summary.json"
        with open(json_path, 'r') as f:
            summary = json.load(f)
        assert summary["accuracy_score"] == pytest.approx(0.75)

    def test_status_is_completed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(metrics, "get_output_path", lambda: str(tmp_path))
        metrics.export_evaluation_summary([0, 1], [0, 1])
        json_path = tmp_path / "evaluation_summary.json"
        with open(json_path, 'r') as f:
            summary = json.load(f)
        assert summary["status"] == "COMPLETED"


class TestLoadBehavioralClassification:

    def test_nonexistent_file_returns_empty_dict(self):
        result = metrics.load_behavioral_classification("/nonexistent/path/file.json")
        assert result == {}

    def test_invalid_json_returns_empty_dict(self):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp.write("NOT VALID JSON {{{")
        tmp.close()
        try:
            result = metrics.load_behavioral_classification(tmp.name)
            assert result == {}
        finally:
            os.unlink(tmp.name)

    def test_missing_persons_key_returns_empty_dict(self):
        path = write_json({"SomeOtherKey": {}})
        try:
            result = metrics.load_behavioral_classification(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_valid_file_returns_persons_dict(self):
        persons_data = {
            "Person_1": {"behavioral_group": "HIGH"},
            "Person_2": {"behavioral_group": "LOW"},
        }
        path = write_json({"persons": persons_data})
        try:
            result = metrics.load_behavioral_classification(path)
            assert result == persons_data
        finally:
            os.unlink(path)

    def test_empty_persons_dict(self):
        path = write_json({"persons": {}})
        try:
            result = metrics.load_behavioral_classification(path)
            assert result == {}
        finally:
            os.unlink(path)