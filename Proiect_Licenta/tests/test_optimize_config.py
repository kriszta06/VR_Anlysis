import numpy as np
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src import optimize_config



def write_csv(lines: list[str]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


def write_json(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def write_config(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def make_scores_json(persons: dict) -> dict:
    return {
        "persons": {
            pid: {"final_score": score}
            for pid, score in persons.items()
        }
    }


SAMPLE_SCORES = {"1": 0.8, "2": 0.3, "3": 0.9, "4": 0.1, "5": 0.6}
SAMPLE_GT = {"1": 1, "2": 0, "3": 1, "4": 0, "5": 1}


class TestLoadGroundTruth:

    def test_valid_csv_returns_correct_mapping(self):
        path = write_csv(["1,1", "2,0", "3,1"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert result == {"1": 1, "2": 0, "3": 1}
        finally:
            os.unlink(path)

    def test_header_row_is_skipped(self):
        path = write_csv(["person_id,diagnosis", "1,1", "2,0"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert "person_id" not in result
            assert result == {"1": 1, "2": 0}
        finally:
            os.unlink(path)

    def test_id_header_variant_is_skipped(self):
        path = write_csv(["id,diagnosis", "1,1"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert "id" not in result
        finally:
            os.unlink(path)

    def test_person_header_variant_is_skipped(self):
        path = write_csv(["person,diagnosis", "1,1"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert "person" not in result
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_none(self):
        result = optimize_config.load_ground_truth("/nonexistent/path/gt.csv")
        assert result is None

    def test_empty_file_returns_none(self):
        path = write_csv([""])
        try:
            result = optimize_config.load_ground_truth(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_all_invalid_labels_returns_none(self):
        path = write_csv(["1,abc", "2,xyz"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_invalid_labels_are_skipped_valid_kept(self):
        path = write_csv(["1,1", "2,bad", "3,0"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert "2" not in result
            assert result["1"] == 1
            assert result["3"] == 0
        finally:
            os.unlink(path)

    def test_lines_with_fewer_than_two_columns_are_skipped(self):
        path = write_csv(["1", "2,0", "3"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert "1" not in result
            assert result == {"2": 0}
        finally:
            os.unlink(path)

    def test_blank_lines_are_ignored(self):
        path = write_csv(["1,1", "", "2,0", ""])
        try:
            result = optimize_config.load_ground_truth(path)
            assert result == {"1": 1, "2": 0}
        finally:
            os.unlink(path)

    def test_returns_dict(self):
        path = write_csv(["1,1"])
        try:
            result = optimize_config.load_ground_truth(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_labels_are_integers(self):
        path = write_csv(["1,1", "2,0"])
        try:
            result = optimize_config.load_ground_truth(path)
            for v in result.values():
                assert isinstance(v, int)
        finally:
            os.unlink(path)


class TestLoadPipelineScores:

    def test_valid_json_returns_correct_scores(self):
        data = make_scores_json({"1": 0.8, "2": 0.3})
        path = write_json(data)
        try:
            result = optimize_config.load_pipeline_scores(path)
            assert result == {"1": 0.8, "2": 0.3}
        finally:
            os.unlink(path)

    def test_person_prefix_is_stripped(self):
        data = make_scores_json({"Person_1": 0.5, "person_2": 0.7})
        path = write_json(data)
        try:
            result = optimize_config.load_pipeline_scores(path)
            assert "1" in result
            assert "2" in result
            assert "Person_1" not in result
        finally:
            os.unlink(path)

    def test_nonexistent_path_returns_none(self):
        result = optimize_config.load_pipeline_scores("/nonexistent/path/scores.json")
        assert result is None

    def test_none_path_falls_back_and_returns_none_when_no_defaults(self):
        with patch("pathlib.Path.exists", return_value=False):
            result = optimize_config.load_pipeline_scores(None)
        assert result is None

    def test_scores_are_floats(self):
        data = make_scores_json({"1": 1, "2": 0})
        path = write_json(data)
        try:
            result = optimize_config.load_pipeline_scores(path)
            for v in result.values():
                assert isinstance(v, float)
        finally:
            os.unlink(path)

    def test_empty_persons_dict_returns_empty_scores(self):
        path = write_json({"persons": {}})
        try:
            result = optimize_config.load_pipeline_scores(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_missing_persons_key_returns_empty_scores(self):
        path = write_json({})
        try:
            result = optimize_config.load_pipeline_scores(path)
            assert result == {}
        finally:
            os.unlink(path)


class TestOptimizeThreshold:

    def test_returns_dict_with_expected_keys(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        assert result is not None
        for key in ("threshold", "f1", "accuracy", "auc", "tp", "tn", "fp", "fn",
                    "n_positive_predicted", "n_positive_true", "persons_evaluated"):
            assert key in result

    def test_no_common_ids_returns_none(self):
        result = optimize_config.optimize_threshold({"99": 0.5}, {"100": 1})
        assert result is None

    def test_persons_evaluated_matches_common_ids(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        assert result["persons_evaluated"] == len(SAMPLE_SCORES)

    def test_n_positive_true_is_correct(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        assert result["n_positive_true"] == sum(SAMPLE_GT.values())

    def test_confusion_matrix_entries_sum_to_n(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        n = result["persons_evaluated"]
        assert result["tp"] + result["tn"] + result["fp"] + result["fn"] == n

    def test_f1_is_between_zero_and_one(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        assert 0.0 <= result["f1"] <= 1.0

    def test_accuracy_is_between_zero_and_one(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_threshold_is_within_score_range(self):
        result = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT)
        scores = list(SAMPLE_SCORES.values())
        assert min(scores) <= result["threshold"] <= max(scores)

    def test_partial_overlap_uses_only_common_ids(self):
        scores = {"1": 0.8, "2": 0.3, "99": 0.5}
        gt = {"1": 1, "2": 0, "100": 1}
        result = optimize_config.optimize_threshold(scores, gt)
        assert result is not None
        assert result["persons_evaluated"] == 2

    def test_all_same_label_auc_is_nan(self):
        scores = {"1": 0.8, "2": 0.3}
        gt = {"1": 1, "2": 1}
        result = optimize_config.optimize_threshold(scores, gt)
        assert result is not None
        assert np.isnan(result["auc"])

    def test_perfect_separation_yields_f1_one(self):
        scores = {"1": 0.9, "2": 0.1, "3": 0.8, "4": 0.2}
        gt = {"1": 1, "2": 0, "3": 1, "4": 0}
        result = optimize_config.optimize_threshold(scores, gt)
        assert result["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_n_steps_parameter_is_respected(self):
        result_coarse = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT, n_steps=10)
        result_fine = optimize_config.optimize_threshold(SAMPLE_SCORES, SAMPLE_GT, n_steps=500)
        assert result_coarse is not None
        assert result_fine is not None

    def test_returns_none_for_empty_dicts(self):
        result = optimize_config.optimize_threshold({}, {})
        assert result is None


class TestOptimizeWeights:

    def test_none_pipeline_fn_returns_none(self):
        result = optimize_config.optimize_weights(None, SAMPLE_GT)
        assert result is None

    def test_returns_dict_with_expected_keys(self):
        def dummy_pipeline(w1, w2, w3):
            return {pid: w1 * s + w2 * s + w3 * s for pid, s in SAMPLE_SCORES.items()}

        result = optimize_config.optimize_weights(dummy_pipeline, SAMPLE_GT, weight_steps=3)
        assert result is not None
        for key in ("weights", "f1", "accuracy", "threshold", "tp", "tn", "fp", "fn"):
            assert key in result

    def test_weights_sum_to_one(self):
        def dummy_pipeline(w1, w2, w3):
            return {pid: w1 * s + w2 * s + w3 * s for pid, s in SAMPLE_SCORES.items()}

        result = optimize_config.optimize_weights(dummy_pipeline, SAMPLE_GT, weight_steps=3)
        w = result["weights"]
        total = w["distance"] + w["mahalanobis"] + w["consistency"]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_pipeline_fn_called_with_valid_weight_triplets(self):
        calls = []

        def tracking_pipeline(w1, w2, w3):
            calls.append((w1, w2, w3))
            return {pid: s for pid, s in SAMPLE_SCORES.items()}

        optimize_config.optimize_weights(tracking_pipeline, SAMPLE_GT, weight_steps=3)
        for w1, w2, w3 in calls:
            assert abs(w1 + w2 + w3 - 1.0) < 1e-5

    def test_pipeline_returning_no_common_ids_is_skipped(self):
        def bad_pipeline(w1, w2, w3):
            return {"999": 0.5}

        result = optimize_config.optimize_weights(bad_pipeline, SAMPLE_GT, weight_steps=3)
        assert result is None or result["weights"] is None


class TestSuggestCategoricalThresholds:

    def test_returns_dict_with_low_medium_high(self):
        result = optimize_config.suggest_categorical_thresholds(SAMPLE_SCORES, SAMPLE_GT, 0.5)
        assert result is not None
        assert set(result.keys()) == {"LOW", "MEDIUM", "HIGH"}

    def test_medium_equals_binary_threshold(self):
        result = optimize_config.suggest_categorical_thresholds(SAMPLE_SCORES, SAMPLE_GT, 0.5)
        assert result["MEDIUM"] == pytest.approx(0.5, abs=1e-4)

    def test_low_is_85_percent_of_threshold(self):
        result = optimize_config.suggest_categorical_thresholds(SAMPLE_SCORES, SAMPLE_GT, 0.5)
        assert result["LOW"] == pytest.approx(0.5 * 0.85, abs=1e-3)

    def test_monotonicity_low_le_medium_le_high(self):
        result = optimize_config.suggest_categorical_thresholds(SAMPLE_SCORES, SAMPLE_GT, 0.5)
        assert result["LOW"] <= result["MEDIUM"] <= result["HIGH"]

    def test_no_positive_scores_returns_none(self):
        gt_all_negative = {"1": 0, "2": 0, "3": 0}
        scores = {"1": 0.5, "2": 0.3, "3": 0.7}
        result = optimize_config.suggest_categorical_thresholds(scores, gt_all_negative, 0.5)
        assert result is None

    def test_fewer_than_four_positives_uses_fallback_high(self):
        scores = {"1": 0.9, "2": 0.3}
        gt = {"1": 1, "2": 0}
        result = optimize_config.suggest_categorical_thresholds(scores, gt, 0.5)
        assert result["HIGH"] == pytest.approx(0.5 * 1.15, abs=1e-3)

    def test_four_or_more_positives_uses_percentile_for_high(self):
        scores = {"1": 0.9, "2": 0.7, "3": 0.8, "4": 0.6, "5": 0.1}
        gt = {"1": 1, "2": 1, "3": 1, "4": 1, "5": 0}
        result = optimize_config.suggest_categorical_thresholds(scores, gt, 0.5)
        expected_high = float(np.percentile([0.9, 0.7, 0.8, 0.6], 75))
        assert result["HIGH"] == pytest.approx(expected_high, abs=1e-3)

    def test_values_are_rounded_to_four_decimal_places(self):
        result = optimize_config.suggest_categorical_thresholds(SAMPLE_SCORES, SAMPLE_GT, 0.123456789)
        for key in ("LOW", "MEDIUM", "HIGH"):
            assert result[key] == round(result[key], 4)

    def test_positive_ids_not_in_scores_are_excluded(self):
        scores = {"1": 0.9}
        gt = {"1": 1, "2": 1}
        result = optimize_config.suggest_categorical_thresholds(scores, gt, 0.5)
        assert result is not None


class TestUpdateConfig:

    SAMPLE_CONFIG = (
        "DISABILITY_THRESHOLD = 0.5\n"
        "DISABILITY_SCORE_THRESHOLDS = {\n"
        "'HIGH': 0.8,\n"
        "'MEDIUM': 0.5,\n"
        "'LOW': 0.3,\n"
        "}\n"
        "DISABILITY_WEIGHTS = {\n"
        "'distance': 0.4,\n"
        "'mahalanobis': 0.4,\n"
        "'consistency': 0.2,\n"
        "}\n"
        "OTHER_SETTING = 42\n"
    )

    def test_returns_true_on_success(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            result = optimize_config.update_config(path, 0.7, {"LOW": 0.4, "MEDIUM": 0.7, "HIGH": 0.9})
            assert result is True
        finally:
            os.unlink(path)

    def test_returns_false_for_nonexistent_file(self):
        result = optimize_config.update_config("/nonexistent/config.py", 0.5, {"LOW": 0.3, "MEDIUM": 0.5, "HIGH": 0.8})
        assert result is False

    def test_threshold_is_updated(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            optimize_config.update_config(path, 0.77, {"LOW": 0.4, "MEDIUM": 0.77, "HIGH": 0.9})
            content = Path(path).read_text(encoding="utf-8")
            assert "DISABILITY_THRESHOLD = 0.77" in content
        finally:
            os.unlink(path)

    def test_categorical_thresholds_are_updated(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            optimize_config.update_config(path, 0.7, {"LOW": 0.55, "MEDIUM": 0.7, "HIGH": 0.95})
            content = Path(path).read_text(encoding="utf-8")
            assert "'LOW': 0.55" in content
            assert "'MEDIUM': 0.7" in content
            assert "'HIGH': 0.95" in content
        finally:
            os.unlink(path)

    def test_weights_are_updated_when_provided(self):
        path = write_config(self.SAMPLE_CONFIG)
        new_weights = {"distance": 0.5, "mahalanobis": 0.3, "consistency": 0.2}
        try:
            optimize_config.update_config(path, 0.7, {"LOW": 0.4, "MEDIUM": 0.7, "HIGH": 0.9}, new_weights=new_weights)
            content = Path(path).read_text(encoding="utf-8")
            assert "'distance': 0.5" in content
            assert "'mahalanobis': 0.3" in content
            assert "'consistency': 0.2" in content
        finally:
            os.unlink(path)

    def test_weights_unchanged_when_not_provided(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            optimize_config.update_config(path, 0.7, {"LOW": 0.4, "MEDIUM": 0.7, "HIGH": 0.9}, new_weights=None)
            content = Path(path).read_text(encoding="utf-8")
            assert "'distance': 0.4" in content
        finally:
            os.unlink(path)

    def test_other_settings_are_preserved(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            optimize_config.update_config(path, 0.7, {"LOW": 0.4, "MEDIUM": 0.7, "HIGH": 0.9})
            content = Path(path).read_text(encoding="utf-8")
            assert "OTHER_SETTING = 42" in content
        finally:
            os.unlink(path)

    def test_old_threshold_value_is_replaced(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            optimize_config.update_config(path, 0.99, {"LOW": 0.4, "MEDIUM": 0.99, "HIGH": 1.1})
            content = Path(path).read_text(encoding="utf-8")
            assert "DISABILITY_THRESHOLD = 0.5" not in content
            assert "DISABILITY_THRESHOLD = 0.99" in content
        finally:
            os.unlink(path)

    def test_file_ends_with_newline(self):
        path = write_config(self.SAMPLE_CONFIG)
        try:
            optimize_config.update_config(path, 0.7, {"LOW": 0.4, "MEDIUM": 0.7, "HIGH": 0.9})
            content = Path(path).read_text(encoding="utf-8")
            assert content.endswith("\n")
        finally:
            os.unlink(path)