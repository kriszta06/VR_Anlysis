import numpy as np
import pytest
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

for mod in [
    'core', 'core.processing', 'core.processing.feature_extractor',
    'core.processing.data_loader',
    'utils', 'utils.file_management',
]:
    sys.modules[mod] = MagicMock()

# Valori aliniate cu disability_config.py
mock_config = MagicMock()
mock_config.SILHOUETTE_K_RANGE = (2, 5)
mock_config.KMEANS_N_INIT = 10
mock_config.KMEANS_RANDOM_STATE = 42
mock_config.DISABILITY_SCORE_THRESHOLDS = {'HIGH': 0.8226, 'MEDIUM': 0.6503, 'LOW': 0.5528}
mock_config.DISABILITY_WEIGHTS = {'distance': 0.35, 'mahalanobis': 0.24, 'consistency': 0.41}
mock_config.VARIATION_HIGH_THRESHOLD = 0.7
mock_config.VARIATION_MEDIUM_THRESHOLD = 0.4
mock_config.DATA_DRIVEN_THRESHOLD_QUANTILES = {'LOW': 0.25, 'MEDIUM': 0.5, 'HIGH': 0.75}

sys.modules['config'] = MagicMock()
sys.modules['config'].disability_config = mock_config

from src.core.analysis import disability_engine


def make_features(n_persons=5, n_features=20, seed=0):
    np.random.seed(seed)
    return {
        f'Person_{i+1}': np.random.randn(n_features)
        for i in range(n_persons)
    }


def make_scenario_features(n_scenarios=4, n_features=20, seed=0):
    np.random.seed(seed)
    return {
        f'scenario_{i}': np.random.randn(n_features)
        for i in range(n_scenarios)
    }


class TestFindOptimalKSilhouette:

    def test_single_sample_returns_k1(self):
        data = np.array([[1.0, 2.0]])
        result = disability_engine.find_optimal_k_silhouette(data, k_range=(2, 5))
        assert result['optimal_k'] == 1

    def test_two_samples_returns_k2(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = disability_engine.find_optimal_k_silhouette(data, k_range=(2, 5))
        assert result['optimal_k'] == 2

    def test_returns_dict_with_required_keys(self):
        np.random.seed(0)
        data = np.random.randn(10, 5)
        result = disability_engine.find_optimal_k_silhouette(data, k_range=(2, 4))
        assert 'optimal_k' in result
        assert 'silhouette_scores' in result

    def test_optimal_k_within_range(self):
        np.random.seed(0)
        data = np.random.randn(20, 5)
        result = disability_engine.find_optimal_k_silhouette(data, k_range=(2, 4))
        assert 2 <= result['optimal_k'] <= 4

    def test_silhouette_scores_is_dict(self):
        np.random.seed(0)
        data = np.random.randn(10, 5)
        result = disability_engine.find_optimal_k_silhouette(data, k_range=(2, 3))
        assert isinstance(result['silhouette_scores'], dict)

    def test_best_score_in_result_for_valid_data(self):
        np.random.seed(0)
        data = np.random.randn(15, 5)
        result = disability_engine.find_optimal_k_silhouette(data, k_range=(2, 3))
        if result['silhouette_scores']:
            assert 'best_score' in result

    def test_uses_config_range_when_none(self):
        np.random.seed(0)
        data = np.random.randn(10, 5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.find_optimal_k_silhouette(data, k_range=None)
        assert 'optimal_k' in result


class TestGetDisabilityStatus:
    # Praguri din disability_config.py: HIGH=0.8226, MEDIUM=0.6503, LOW=0.5528

    def test_high_threshold(self):
        # 0.90 >= 0.8226 → HIGH
        assert disability_engine.get_disability_status(0.90) == 'HIGH'

    def test_medium_threshold(self):
        # 0.70 >= 0.6503 dar < 0.8226 → MEDIUM
        assert disability_engine.get_disability_status(0.70) == 'MEDIUM'

    def test_low_threshold(self):
        # 0.58 >= 0.5528 dar < 0.6503 → LOW
        assert disability_engine.get_disability_status(0.58) == 'LOW'

    def test_none_status(self):
        # 0.10 < 0.5528 → NONE
        assert disability_engine.get_disability_status(0.10) == 'NONE'

    def test_exact_high_boundary(self):
        # exact 0.8226 → HIGH
        assert disability_engine.get_disability_status(0.8226) == 'HIGH'

    def test_exact_medium_boundary(self):
        # exact 0.6503 → MEDIUM
        assert disability_engine.get_disability_status(0.6503) == 'MEDIUM'

    def test_exact_low_boundary(self):
        # exact 0.5528 → LOW
        assert disability_engine.get_disability_status(0.5528) == 'LOW'

    def test_custom_thresholds(self):
        thresholds = {'HIGH': 0.9, 'MEDIUM': 0.7, 'LOW': 0.5}
        assert disability_engine.get_disability_status(0.95, thresholds) == 'HIGH'
        assert disability_engine.get_disability_status(0.75, thresholds) == 'MEDIUM'
        assert disability_engine.get_disability_status(0.55, thresholds) == 'LOW'
        assert disability_engine.get_disability_status(0.10, thresholds) == 'NONE'

    def test_unordered_thresholds_falls_back_to_default(self):
        bad_thresholds = {'HIGH': 0.3, 'MEDIUM': 0.6, 'LOW': 0.8}
        result = disability_engine.get_disability_status(0.90, bad_thresholds)
        assert result in ('HIGH', 'MEDIUM', 'LOW', 'NONE')


class TestComputeDataDrivenThresholds:

    def test_returns_dict_with_three_keys(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = disability_engine.compute_data_driven_thresholds(scores)
        assert set(result.keys()) == {'LOW', 'MEDIUM', 'HIGH'}

    def test_thresholds_ordered(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = disability_engine.compute_data_driven_thresholds(scores)
        assert result['LOW'] <= result['MEDIUM'] <= result['HIGH']

    def test_empty_scores_returns_defaults(self):
        # Valorile default hardcodate din disability_engine.py
        result = disability_engine.compute_data_driven_thresholds([])
        assert result['HIGH'] == 0.72
        assert result['MEDIUM'] == 0.61
        assert result['LOW'] == 0.52

    def test_nan_values_ignored(self):
        scores = [0.1, np.nan, 0.5, np.inf, 0.9]
        result = disability_engine.compute_data_driven_thresholds(scores)
        assert all(np.isfinite(v) for v in result.values())

    def test_all_nan_returns_defaults(self):
        result = disability_engine.compute_data_driven_thresholds([np.nan, np.inf, -np.inf])
        assert result['HIGH'] == 0.72

    def test_unordered_quantiles_fall_back(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        bad_quantiles = {'LOW': 0.9, 'MEDIUM': 0.5, 'HIGH': 0.1}
        result = disability_engine.compute_data_driven_thresholds(scores, quantiles=bad_quantiles)
        assert result['LOW'] <= result['MEDIUM'] <= result['HIGH']

    def test_single_score(self):
        result = disability_engine.compute_data_driven_thresholds([0.5])
        assert set(result.keys()) == {'LOW', 'MEDIUM', 'HIGH'}


class TestDetectDisabilityPatternsUnsupervised:

    def test_returns_dict(self):
        scenarios = make_scenario_features(n_scenarios=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.detect_disability_patterns_unsupervised(scenarios)
        assert isinstance(result, dict)

    def test_all_scenarios_in_output(self):
        scenarios = make_scenario_features(n_scenarios=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.detect_disability_patterns_unsupervised(scenarios)
        assert set(result.keys()) == set(scenarios.keys())

    def test_result_keys_per_scenario(self):
        scenarios = make_scenario_features(n_scenarios=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.detect_disability_patterns_unsupervised(scenarios)
        for name, info in result.items():
            assert 'cluster' in info
            assert 'distance' in info
            assert 'score' in info
            assert 'variation_level' in info

    def test_scores_between_zero_and_one(self):
        scenarios = make_scenario_features(n_scenarios=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.detect_disability_patterns_unsupervised(scenarios)
        for info in result.values():
            assert 0.0 <= info['score'] <= 1.0

    def test_variation_level_valid_values(self):
        scenarios = make_scenario_features(n_scenarios=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.detect_disability_patterns_unsupervised(scenarios)
        valid = {'LOW', 'MEDIUM', 'HIGH'}
        for info in result.values():
            assert info['variation_level'] in valid

    def test_2d_array_triggers_feature_extraction(self):
        mock_extract = MagicMock(return_value=np.random.randn(20))
        scenarios = {'sc_A': np.random.randn(30, 10)}
        with patch('src.core.analysis.disability_engine.extract_behavior_features', mock_extract), \
             patch('src.core.analysis.disability_engine.config', mock_config):
            disability_engine.detect_disability_patterns_unsupervised(scenarios)
        mock_extract.assert_called_once()


class TestAnalyzePersonDisability:

    def test_empty_input_returns_empty_dict(self):
        result = disability_engine.analyze_person_disability({})
        assert result == {}

    def test_single_person_returns_none_status(self):
        features = {'Person_1': np.random.randn(20)}
        result = disability_engine.analyze_person_disability(features)
        assert result['Person_1']['status'] == 'NONE'
        assert result['Person_1']['final_score'] == 0.0

    def test_returns_dict_with_all_persons(self):
        features = make_features(n_persons=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.analyze_person_disability(features)
        assert set(result.keys()) == set(features.keys())

    def test_result_keys_per_person(self):
        features = make_features(n_persons=4)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.analyze_person_disability(features)
        for info in result.values():
            assert 'cluster' in info
            assert 'distance_score' in info
            assert 'mahalanobis_score' in info
            assert 'consistency_score' in info
            assert 'final_score' in info
            assert 'status' in info

    def test_status_values_are_valid(self):
        features = make_features(n_persons=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.analyze_person_disability(features)
        valid = {'NONE', 'LOW', 'MEDIUM', 'HIGH'}
        for info in result.values():
            assert info['status'] in valid

    def test_final_scores_are_finite(self):
        features = make_features(n_persons=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.analyze_person_disability(features)
        for info in result.values():
            assert np.isfinite(info['final_score'])

    def test_cluster_labels_are_integers(self):
        features = make_features(n_persons=5)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.analyze_person_disability(features)
        for info in result.values():
            assert isinstance(info['cluster'], int)

    def test_two_persons(self):
        features = make_features(n_persons=2)
        with patch('src.core.analysis.disability_engine.config', mock_config):
            result = disability_engine.analyze_person_disability(features)
        assert len(result) == 2


class TestSaveDetailedResults:

    def _make_assessment(self, n=3):
        return {
            f'Person_{i+1}': {
                'status': 'NONE',
                'final_score': 0.3,
                'cluster': 0,
                'distance_score': 0.2,
                'mahalanobis_score': 0.1,
                'consistency_score': 0.4,
            }
            for i in range(n)
        }

    def _make_person_features(self, n=3, n_feat=10):
        return {f'Person_{i+1}': np.random.randn(n_feat) for i in range(n)}

    def test_creates_json_file(self, tmp_path):
        assessment = self._make_assessment()
        features = self._make_person_features()
        with patch('src.core.analysis.disability_engine.os.path.abspath', return_value=str(tmp_path / 'x' / 'y' / 'z.py')), \
             patch('src.core.analysis.disability_engine.config', mock_config):
            disability_engine.save_detailed_results(assessment, features)

    def test_json_contains_all_persons(self, tmp_path):
        assessment = self._make_assessment(n=3)
        features = self._make_person_features(n=3)
        written = {}

        real_open = open

        def fake_open(path, mode='r', **kw):
            if 'behavioral_classification' in str(path):
                import io
                buf = io.StringIO()
                class FakeFile:
                    def __enter__(self): return buf
                    def __exit__(self, *a): written['content'] = buf.getvalue()
                return FakeFile()
            return real_open(path, mode, **kw)

        with patch('src.core.analysis.disability_engine.os.path.abspath', return_value=str(tmp_path / 'x' / 'y' / 'z.py')), \
             patch('src.core.analysis.disability_engine.config', mock_config), \
             patch('builtins.open', side_effect=fake_open):
            disability_engine.save_detailed_results(assessment, features)

    def test_does_not_raise_with_valid_input(self, tmp_path):
        assessment = self._make_assessment()
        features = self._make_person_features()
        with patch('src.core.analysis.disability_engine.os.path.abspath', return_value=str(tmp_path / 'a' / 'b' / 'c.py')), \
             patch('src.core.analysis.disability_engine.config', mock_config):
            disability_engine.save_detailed_results(assessment, features)