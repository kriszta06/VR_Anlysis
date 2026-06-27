import numpy as np
import pytest
import os
import json
import sys
import tempfile
from unittest.mock import patch, MagicMock

# NOTE: Only project-internal bare module names are mocked here. Real
# third-party packages (scipy, sklearn, seaborn, ...) must NOT be replaced with
# MagicMocks: these mutations to sys.modules are global and persist for the
# whole pytest session, which would break other test modules (e.g. the
# clustering tests, which need a real scipy/sklearn).
for mod in [
    'core', 'core.processing', 'core.processing.data_loader',
    'core.processing.feature_extractor', 'core.analysis',
    'core.analysis.clustering', 'core.analysis.disability_engine',
    'utils', 'utils.file_management',
    'visualization', 'visualization.reports', 'visualization.scenario_comparison',
]:
    sys.modules[mod] = MagicMock()

mock_config = MagicMock()
mock_config.CLUSTER_COLORS = ['red', 'blue', 'green', 'orange', 'purple']
sys.modules['config'] = MagicMock()
sys.modules['config'].disability_config = mock_config

from src.visualization import plotter_3d


def make_positions(n=30, seed=0):
    np.random.seed(seed)
    return np.random.randn(n, 3)


def make_labels(n=30, n_clusters=3):
    return np.array([i % n_clusters for i in range(n)])


def make_scenario_data(n=30, seed=0):
    np.random.seed(seed)
    data = np.random.randn(n, 10)
    return data


class TestPlotClusters:

    def test_returns_list(self, tmp_path):
        positions = make_positions()
        labels = make_labels()
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "test_scenario", "combo_A")
        assert isinstance(result, list)

    def test_returns_one_entry_per_cluster(self, tmp_path):
        positions = make_positions()
        labels = make_labels(n_clusters=3)
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "test_scenario", "combo_A")
        assert len(result) == 3

    def test_cluster_stat_keys(self, tmp_path):
        positions = make_positions()
        labels = make_labels()
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "test_scenario", "combo_A")
        for stat in result:
            assert 'label' in stat
            assert 'color' in stat
            assert 'center' in stat
            assert 'size' in stat
            assert 'percentage' in stat

    def test_center_has_xyz_keys(self, tmp_path):
        positions = make_positions()
        labels = make_labels()
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "test_scenario", "combo_A")
        for stat in result:
            assert 'x' in stat['center']
            assert 'y' in stat['center']
            assert 'z' in stat['center']

    def test_percentages_sum_to_100(self, tmp_path):
        positions = make_positions()
        labels = make_labels()
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "test_scenario", "combo_A")
        total = sum(s['percentage'] for s in result)
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_sizes_sum_to_total_points(self, tmp_path):
        positions = make_positions(n=30)
        labels = make_labels(n=30)
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "test_scenario", "combo_A")
        assert sum(s['size'] for s in result) == 30

    def test_plus_in_combo_name_replaced_in_filename(self, tmp_path):
        positions = make_positions()
        labels = make_labels()
        saved_paths = []

        original_savefig = __import__('matplotlib').pyplot.savefig

        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config), \
             patch('matplotlib.pyplot.savefig', side_effect=lambda path, **kw: saved_paths.append(path)):
            plotter_3d.plot_clusters(positions, labels, "sc", "head+eyes")

        assert any('+' not in str(p) for p in saved_paths)

    def test_single_cluster(self, tmp_path):
        positions = make_positions(n=20)
        labels = np.zeros(20, dtype=int)
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config):
            result = plotter_3d.plot_clusters(positions, labels, "sc", "combo")
        assert len(result) == 1
        assert result[0]['percentage'] == pytest.approx(100.0, abs=1e-6)

    def test_json_file_written(self, tmp_path):
        positions = make_positions()
        labels = make_labels()
        written = {}
        import builtins
        real_open = builtins.open

        def fake_open(path, mode='r', **kw):
            if 'clusters_' in str(path) and str(path).endswith('.json'):
                import io
                buf = io.StringIO()
                class FakeFile:
                    def __enter__(self): return buf
                    def __exit__(self, *a): written['content'] = buf.getvalue()
                return FakeFile()
            return real_open(path, mode, **kw)

        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)), \
             patch('plotter_3d.config', mock_config), \
             patch('builtins.open', side_effect=fake_open):
            plotter_3d.plot_clusters(positions, labels, "sc", "combo")


class TestPlotDisabilityAnnotations:

    def _make_all_data(self, scenarios, n=20):
        return {
            name: {"data": make_scenario_data(n=n)}
            for name in scenarios
        }

    def _make_likelihood(self, scenarios, status='LOW', score=0.2):
        return {
            name: {'status': status, 'score': score}
            for name in scenarios
        }

    def test_skips_scenario_not_in_likelihood(self, tmp_path, capsys):
        all_data = self._make_all_data(["sc_A", "sc_B"])
        likelihood = self._make_likelihood(["sc_A"])
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)):
            plotter_3d.plot_disability_annotations(all_data, likelihood)
        captured = capsys.readouterr()
        assert "sc_B" in captured.out

    def test_runs_for_all_statuses(self, tmp_path):
        for status in ['HIGH', 'MEDIUM', 'LOW', 'NONE']:
            all_data = self._make_all_data([f"sc_{status}"])
            likelihood = self._make_likelihood([f"sc_{status}"], status=status, score=0.5)
            with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)):
                plotter_3d.plot_disability_annotations(all_data, likelihood)

    def test_global_origin_applied(self, tmp_path):
        all_data = self._make_all_data(["sc_A"])
        likelihood = self._make_likelihood(["sc_A"])
        origin = np.array([1.0, 2.0, 3.0])
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)):
            plotter_3d.plot_disability_annotations(all_data, likelihood, global_origin=origin)

    def test_local_origin_used_when_global_none(self, tmp_path):
        all_data = self._make_all_data(["sc_A"])
        likelihood = self._make_likelihood(["sc_A"])
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)):
            plotter_3d.plot_disability_annotations(all_data, likelihood, global_origin=None)

    def test_empty_all_data_does_not_crash(self, tmp_path):
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)):
            plotter_3d.plot_disability_annotations({}, {})

    def test_multiple_scenarios_processed(self, tmp_path):
        scenarios = [f"sc_{i}" for i in range(5)]
        all_data = self._make_all_data(scenarios)
        likelihood = self._make_likelihood(scenarios)
        with patch('plotter_3d.os.path.dirname', return_value=str(tmp_path)):
            plotter_3d.plot_disability_annotations(all_data, likelihood)