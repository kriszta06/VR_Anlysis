import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from src import mann_whitney



def write_json(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def make_sample_results(n_healthy=5, n_disabled=5):
    return {
        "n_healthy": n_healthy,
        "n_disabled": n_disabled,
        "u_statistic": 20.0,
        "u_max": float(n_healthy * n_disabled),
        "p_value_two_sided": 0.03,
        "p_value_one_sided": 0.015,
        "effect_size_r": 0.45,
        "effect_size_ci_95": [0.10, 0.70],
        "probability_superiority": 0.8,
        "direction": "atypical > typical",
        "effect_label": "MEDIUM",
        "effect_description": "Difference between groups is medium",
        "significance": "Significant (p < 0.05)",
        "reject_h0_alpha05": True,
        "reject_h0_alpha10": True,
        "median_healthy": 0.2,
        "median_disabled": 0.6,
        "mean_healthy": 0.25,
        "mean_disabled": 0.62,
        "std_healthy": 0.05,
        "std_disabled": 0.08,
        "boot_r_distribution": [0.4, 0.45, 0.5],
    }


def make_per_person(n_healthy=5, n_disabled=5):
    per_person = []
    for i in range(n_healthy):
        per_person.append({"person": f"Person_{i}", "true_label": 0, "final_score": 0.1 + 0.01 * i})
    for i in range(n_disabled):
        per_person.append({"person": f"Person_{100 + i}", "true_label": 1, "final_score": 0.6 + 0.01 * i})
    return per_person


class TestComputeMannWhitney:

    def test_returns_dict_with_expected_keys(self):
        healthy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        disabled = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        result = mann_whitney.compute_mann_whitney(healthy, disabled)
        expected_keys = {
            "n_healthy", "n_disabled", "u_statistic", "u_max",
            "p_value_two_sided", "p_value_one_sided", "effect_size_r",
            "effect_size_ci_95", "probability_superiority", "direction",
            "effect_label", "effect_description", "significance",
            "reject_h0_alpha05", "reject_h0_alpha10", "median_healthy",
            "median_disabled", "mean_healthy", "mean_disabled",
            "std_healthy", "std_disabled", "boot_r_distribution",
        }
        assert expected_keys.issubset(result.keys())

    def test_group_sizes_correct(self):
        healthy = np.array([1.0, 2.0, 3.0])
        disabled = np.array([4.0, 5.0])
        result = mann_whitney.compute_mann_whitney(healthy, disabled)
        assert result["n_healthy"] == 3
        assert result["n_disabled"] == 2
        assert result["u_max"] == 6.0

    def test_complete_separation_gives_major_effect_and_low_p(self):
        healthy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        disabled = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = mann_whitney.compute_mann_whitney(healthy, disabled)
        assert result["effect_label"] == "MAJOR"
        assert result["direction"] == "atypical > typical"
        assert result["p_value_two_sided"] < 0.01
        assert result["significance"] == "Highly significant (p < 0.01)"
        assert result["reject_h0_alpha05"] is True

    def test_identical_distributions_give_minor_effect_and_high_p(self):
        healthy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        disabled = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mann_whitney.compute_mann_whitney(healthy, disabled)
        assert result["effect_label"] == "MINOR"
        assert result["p_value_two_sided"] > 0.10
        assert result["significance"] == "Not significant (p \u2265 0.10)"
        assert result["reject_h0_alpha05"] is False
        assert result["reject_h0_alpha10"] is False

    def test_boot_distribution_truncated_to_200(self):
        healthy = np.array([1.0, 2.0, 3.0, 4.0])
        disabled = np.array([5.0, 6.0, 7.0, 8.0])
        result = mann_whitney.compute_mann_whitney(healthy, disabled)
        assert len(result["boot_r_distribution"]) == 200

    def test_deterministic_with_fixed_seed(self):
        healthy = np.array([1.0, 2.0, 3.0, 4.0])
        disabled = np.array([5.0, 6.0, 7.0, 8.0])
        result_a = mann_whitney.compute_mann_whitney(healthy, disabled)
        result_b = mann_whitney.compute_mann_whitney(healthy, disabled)
        assert result_a["effect_size_ci_95"] == result_b["effect_size_ci_95"]
        assert result_a["boot_r_distribution"] == result_b["boot_r_distribution"]


class TestGeneratePlots:

    def test_creates_plot_file(self, tmp_path):
        per_person = make_per_person()
        results = make_sample_results()
        out_path = mann_whitney.generate_plots(per_person, results, output_dir=str(tmp_path))
        assert os.path.exists(out_path)
        assert out_path.endswith("mann_whitney_plot.png")

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested_dir = tmp_path / "nested" / "plots"
        per_person = make_per_person()
        results = make_sample_results()
        mann_whitney.generate_plots(per_person, results, output_dir=str(nested_dir))
        assert nested_dir.exists()

    def test_returns_string_path(self, tmp_path):
        per_person = make_per_person()
        results = make_sample_results()
        out_path = mann_whitney.generate_plots(per_person, results, output_dir=str(tmp_path))
        assert isinstance(out_path, str)


class TestPrintSummary:

    def test_prints_header_and_core_stats(self, capsys):
        results = make_sample_results()
        mann_whitney.print_summary(results)
        captured = capsys.readouterr()
        assert "MANN-WHITNEY U TEST" in captured.out
        assert "U statistic" in captured.out
        assert "Effect size r" in captured.out

    def test_alpha05_rejection_message(self, capsys):
        results = make_sample_results()
        results["reject_h0_alpha05"] = True
        results["reject_h0_alpha10"] = True
        mann_whitney.print_summary(results)
        captured = capsys.readouterr()
        assert "H0 rejected at \u03b1 = 0.05" in captured.out

    def test_alpha10_only_rejection_message(self, capsys):
        results = make_sample_results()
        results["reject_h0_alpha05"] = False
        results["reject_h0_alpha10"] = True
        mann_whitney.print_summary(results)
        captured = capsys.readouterr()
        assert "marginal" in captured.out

    def test_no_rejection_message(self, capsys):
        results = make_sample_results()
        results["reject_h0_alpha05"] = False
        results["reject_h0_alpha10"] = False
        mann_whitney.print_summary(results)
        captured = capsys.readouterr()
        assert "H0 cannot be rejected" in captured.out


class TestRunMannWhitney:

    def test_missing_json_file_returns_none(self, tmp_path):
        result = mann_whitney.run_mann_whitney(
            person_scores=None,
            ground_truth_path=str(tmp_path / "missing_gt.csv"),
            json_path=str(tmp_path / "missing.json"),
            output_dir=str(tmp_path),
        )
        assert result is None

    def test_missing_ground_truth_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mann_whitney, "load_ground_truth", lambda path: {})
        person_scores = {f"{i}": 0.1 * i for i in range(5)}
        result = mann_whitney.run_mann_whitney(
            person_scores=person_scores,
            ground_truth_path="ignored",
            output_dir=str(tmp_path),
        )
        assert result is None

    def test_fewer_than_two_disabled_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            mann_whitney, "load_ground_truth",
            lambda path: {"0": 0, "1": 0, "2": 0, "3": 1},
        )
        person_scores = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.9}
        result = mann_whitney.run_mann_whitney(
            person_scores=person_scores,
            ground_truth_path="ignored",
            output_dir=str(tmp_path),
        )
        assert result is None

    def test_successful_integrated_mode_saves_results(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            mann_whitney, "load_ground_truth",
            lambda path: {str(i): int(i >= 5) for i in range(10)},
        )
        monkeypatch.setattr(mann_whitney, "generate_plots", lambda *a, **k: str(tmp_path / "plot.png"))

        person_scores = {str(i): (0.1 + 0.01 * i if i < 5 else 0.7 + 0.01 * i) for i in range(10)}
        result = mann_whitney.run_mann_whitney(
            person_scores=person_scores,
            ground_truth_path="ignored",
            output_dir=str(tmp_path),
        )
        assert result is not None
        out_json = tmp_path / "mann_whitney_results.json"
        assert out_json.exists()
        with open(out_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert "boot_r_distribution" not in saved
        assert "per_person_scores" in saved

    def test_standalone_mode_normalizes_person_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            mann_whitney, "load_ground_truth",
            lambda path: {str(i): int(i >= 5) for i in range(10)},
        )
        monkeypatch.setattr(mann_whitney, "generate_plots", lambda *a, **k: str(tmp_path / "plot.png"))

        persons = {
            f"Person_{i}": {"final_score": (0.1 + 0.01 * i if i < 5 else 0.7 + 0.01 * i)}
            for i in range(10)
        }
        json_path = write_json({"persons": persons})
        try:
            result = mann_whitney.run_mann_whitney(
                person_scores=None,
                ground_truth_path="ignored",
                json_path=json_path,
                output_dir=str(tmp_path),
            )
            assert result is not None
        finally:
            os.unlink(json_path)