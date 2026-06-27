import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sys

mock_config = MagicMock()
mock_config.FEATURES_ENTROPY_BINS = 5
mock_config.FEATURES_WINDOW_SIZE = 5
mock_config.FEATURES_PAUSE_BASE_THRESHOLD = 0.01
mock_config.FEATURES_SHARP_TURN_DEGREES = 30
mock_config.FEATURES_AUTOCORR_MIN_FRAMES = 5
mock_config.FEATURES_AUTOCORR_MAX_FRAMES = 10

sys.modules['config'] = mock_config
sys.modules['core'] = MagicMock()
sys.modules['core.processing'] = MagicMock()
sys.modules['core.processing.data_loader'] = MagicMock()

from src.core.processing import feature_extractor

def make_scenario(n_frames=50, seed=42):
    np.random.seed(seed)
    positions  = np.cumsum(np.random.randn(n_frames, 3) * 0.1, axis=0)
    rotations  = np.random.randn(n_frames, 3) * 5
    forwards   = np.tile([0.0, 0.0, 1.0], (n_frames, 1))
    timestamps = np.linspace(0.0, 10.0, n_frames).reshape(-1, 1)
    return np.hstack([positions, rotations, forwards, timestamps])


def make_stationary(n_frames=50):
    positions  = np.zeros((n_frames, 3))
    rotations  = np.zeros((n_frames, 3))
    forwards   = np.tile([0.0, 0.0, 1.0], (n_frames, 1))
    timestamps = np.linspace(0.0, 10.0, n_frames).reshape(-1, 1)
    return np.hstack([positions, rotations, forwards, timestamps])


def make_circular(n_frames=50):
    angles     = np.linspace(0, 2 * np.pi, n_frames)
    positions  = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_frames)]) * 2
    rotations  = np.zeros((n_frames, 3))
    forwards   = np.tile([0.0, 0.0, 1.0], (n_frames, 1))
    timestamps = np.linspace(0.0, 10.0, n_frames).reshape(-1, 1)
    return np.hstack([positions, rotations, forwards, timestamps])


class TestExtractBehaviorFeaturesOutput:

    def test_returns_numpy_array(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        assert isinstance(result, np.ndarray)

    def test_output_is_one_dimensional(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        assert result.ndim == 1

    def test_output_length_is_consistent(self):
        r1 = feature_extractor.extract_behavior_features(make_scenario(n_frames=50, seed=0))
        r2 = feature_extractor.extract_behavior_features(make_scenario(n_frames=80, seed=1))
        assert len(r1) == len(r2)

    def test_no_nan_in_output(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        assert not np.any(np.isnan(result))

    def test_no_inf_in_output(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        assert not np.any(np.isinf(result))

    def test_output_is_finite(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        assert np.all(np.isfinite(result))


class TestPositionFeatures:

    def test_total_distance_greater_than_zero_for_moving(self):
        data = make_scenario()
        result = feature_extractor.extract_behavior_features(data)
        total_distance = result[6]
        assert total_distance > 0

    def test_total_distance_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        total_distance = result[6]
        assert total_distance == pytest.approx(0.0, abs=1e-6)

    def test_mean_position_correct(self):
        data = make_scenario()
        positions = data[:, :3]
        expected_mean = np.mean(positions, axis=0)
        result = feature_extractor.extract_behavior_features(data)
        np.testing.assert_array_almost_equal(result[:3], expected_mean, decimal=5)

    def test_std_position_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        std_pos = result[3:6]
        np.testing.assert_array_almost_equal(std_pos, [0.0, 0.0, 0.0], decimal=6)

    def test_path_efficiency_between_zero_and_one(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        path_eff_idx = 37
        assert 0.0 <= result[path_eff_idx] <= 1.0

    def test_circular_path_efficiency_near_zero(self):
        result = feature_extractor.extract_behavior_features(make_circular())
        path_eff_idx = 37
        assert result[path_eff_idx] < 0.1


class TestSpeedFeatures:

    def test_mean_speed_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        mean_speed_idx = 22
        assert result[mean_speed_idx] == pytest.approx(0.0, abs=1e-6)

    def test_mean_speed_positive_for_moving(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        mean_speed_idx = 22
        assert result[mean_speed_idx] > 0

    def test_max_speed_geq_mean_speed(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        mean_speed_idx = 22
        max_speed_idx  = 24
        assert result[max_speed_idx] >= result[mean_speed_idx]

    def test_std_speed_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        std_speed_idx = 23
        assert result[std_speed_idx] == pytest.approx(0.0, abs=1e-6)

    def test_pause_ratio_between_zero_and_one(self):
        result = feature_extractor.extract_behavior_features(make_scenario())
        pause_ratio_idx = 43
        assert 0.0 <= result[pause_ratio_idx] <= 1.0

    def test_pause_ratio_high_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        pause_ratio_idx = 43
        assert result[pause_ratio_idx] > 0.5


class TestRotationFeatures:

    def test_mean_rotation_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        mean_rot = result[7:10]
        np.testing.assert_array_almost_equal(mean_rot, [0.0, 0.0, 0.0], decimal=6)

    def test_std_rotation_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        std_rot = result[10:13]
        np.testing.assert_array_almost_equal(std_rot, [0.0, 0.0, 0.0], decimal=6)

    def test_ptp_rotation_zero_for_stationary(self):
        result = feature_extractor.extract_behavior_features(make_stationary())
        ptp_rot_idx = slice(28, 31)
        np.testing.assert_array_almost_equal(result[ptp_rot_idx], [0.0, 0.0, 0.0], decimal=6)


class TestEdgeCases:

    def test_minimal_frames(self):
        data = make_scenario(n_frames=3)
        result = feature_extractor.extract_behavior_features(data)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_single_frame_does_not_crash(self):
        data = make_scenario(n_frames=1)
        try:
            result = feature_extractor.extract_behavior_features(data)
            assert isinstance(result, np.ndarray)
        except Exception as e:
            pytest.fail(f"extract_behavior_features raised an exception with 1 frame: {e}")

    def test_high_rotation_scenario(self):
        n = 50
        positions  = np.cumsum(np.random.randn(n, 3) * 0.05, axis=0)
        rotations  = np.random.randn(n, 3) * 45
        forwards   = np.tile([0.0, 0.0, 1.0], (n, 1))
        timestamps = np.linspace(0.0, 10.0, n).reshape(-1, 1)
        data = np.hstack([positions, rotations, forwards, timestamps])
        result = feature_extractor.extract_behavior_features(data)
        assert np.all(np.isfinite(result))

    def test_different_seeds_produce_different_outputs(self):
        r1 = feature_extractor.extract_behavior_features(make_scenario(seed=0))
        r2 = feature_extractor.extract_behavior_features(make_scenario(seed=99))
        assert not np.allclose(r1, r2)

    def test_deterministic_output_for_same_input(self):
        data = make_scenario(seed=7)
        r1 = feature_extractor.extract_behavior_features(data)
        r2 = feature_extractor.extract_behavior_features(data)
        np.testing.assert_array_equal(r1, r2)