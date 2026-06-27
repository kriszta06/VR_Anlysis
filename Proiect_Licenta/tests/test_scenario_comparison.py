import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

for mod in [
    'core', 'core.processing', 'core.processing.data_loader',
    'utils', 'utils.file_management',
    'mpl_toolkits', 'mpl_toolkits.mplot3d',
]:
    sys.modules[mod] = MagicMock()

from src.visualization import scenario_comparison

# The patches below target the bare module name ``scenario_comparison``.
# conftest puts ``src/visualization`` on sys.path, so a bare ``import
# scenario_comparison`` would create a *second*, distinct module object and the
# patches would not affect the module actually under test. Alias the imported
# module under the bare name so ``patch('scenario_comparison.x')`` resolves to
# this same object.
sys.modules['scenario_comparison'] = scenario_comparison


def make_trajectory(n=30, seed=0):
    np.random.seed(seed)
    return np.random.randn(n, 10)


def make_all_data(persons=None, scenarios=('A', 'B', 'C')):
    if persons is None:
        persons = ['Person_1', 'Person_2']
    return {
        person: {
            f"{i+1}-{s}": make_trajectory(seed=i * 10 + j)
            for j, s in enumerate(scenarios)
        }
        for i, person in enumerate(persons)
    }


class TestPlotAllScenariosComparison:

    def test_empty_data_returns_early(self, capsys):
        scenario_comparison.plot_all_scenarios_comparison({})
        captured = capsys.readouterr()
        assert "No data" in captured.out

    def test_runs_without_exception(self, tmp_path):
        data = make_all_data()
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_all_scenarios_comparison(data)

    def test_runs_with_global_origin(self, tmp_path):
        data = make_all_data()
        origin = np.array([1.0, 2.0, 3.0])
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_all_scenarios_comparison(data, global_origin=origin)

    def test_only_abc_scenario_types_processed(self, tmp_path, capsys):
        data = {
            'Person_1': {
                '1-A': make_trajectory(),
                '1-X': make_trajectory(),
            }
        }
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_all_scenarios_comparison(data)
        captured = capsys.readouterr()
        assert 'X' not in captured.out or 'comparison' not in captured.out

    def test_missing_scenario_type_skipped(self, tmp_path, capsys):
        data = {'Person_1': {'1-A': make_trajectory()}}
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_all_scenarios_comparison(data)
        captured = capsys.readouterr()
        assert 'A' in captured.out

    def test_multiple_persons_all_plotted(self, tmp_path, capsys):
        data = make_all_data(persons=['Person_1', 'Person_2', 'Person_3'])
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_all_scenarios_comparison(data)
        captured = capsys.readouterr()
        assert 'A' in captured.out
        assert 'B' in captured.out
        assert 'C' in captured.out

    def test_returns_none(self, tmp_path):
        data = make_all_data()
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            result = scenario_comparison.plot_all_scenarios_comparison(data)
        assert result is None


class TestPlotPersonScenarioComparison:

    def test_empty_data_returns_early(self, capsys):
        scenario_comparison.plot_person_scenario_comparison({}, 'Person_1')
        captured = capsys.readouterr()
        assert "No data" in captured.out

    def test_runs_without_exception(self, tmp_path):
        data = {'1-A': make_trajectory(), '1-B': make_trajectory()}
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_person_scenario_comparison(data, 'Person_1')

    def test_runs_with_global_origin(self, tmp_path):
        data = {'1-A': make_trajectory()}
        origin = np.array([0.5, 1.0, 1.5])
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_person_scenario_comparison(data, 'Person_1', global_origin=origin)

    def test_local_origin_used_when_global_none(self, tmp_path):
        data = {'1-A': make_trajectory()}
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_person_scenario_comparison(data, 'Person_1', global_origin=None)

    def test_output_message_contains_person_id(self, tmp_path, capsys):
        data = {'1-A': make_trajectory()}
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_person_scenario_comparison(data, 'Person_42')
        captured = capsys.readouterr()
        assert 'Person_42' in captured.out

    def test_single_scenario_does_not_crash(self, tmp_path):
        data = {'1-A': make_trajectory()}
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            scenario_comparison.plot_person_scenario_comparison(data, 'Person_1')

    def test_returns_none(self, tmp_path):
        data = {'1-A': make_trajectory()}
        with patch('scenario_comparison.os.path.dirname', return_value=str(tmp_path)):
            result = scenario_comparison.plot_person_scenario_comparison(data, 'Person_1')
        assert result is None


class TestLoadScenarioPositionsForPersons:

    def _make_json_files(self, names):
        return [Path(f"data/vr_recordings/{name}") for name in names]

    def test_returns_dict(self):
        mock_group = {'Person_1': [Path('1-A.json')]}
        mock_positions = np.random.randn(20, 3)
        mock_timestamps = np.linspace(0, 10, 20)

        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(mock_positions, None, None, mock_timestamps)):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1'], [])
        assert isinstance(result, dict)

    def test_person_not_in_grouped_skipped(self, capsys):
        with patch('scenario_comparison.group_files_by_person', return_value={}):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_99'], [])
        assert 'Person_99' not in result
        assert 'Person_99' in capsys.readouterr().out

    def test_empty_timestamps_skips_scenario(self):
        mock_group = {'Person_1': [Path('1-A.json')]}
        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(np.array([]), None, None, np.array([]))):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1'], [])
        assert 'Person_1' not in result

    def test_too_much_padding_skips_scenario(self):
        mock_group = {'Person_1': [Path('1-A.json')]}
        padded = np.zeros((20, 3))
        timestamps = np.linspace(0, 10, 20)
        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(padded, None, None, timestamps)):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1'], [])
        assert 'Person_1' not in result

    def test_valid_data_included(self):
        mock_group = {'Person_1': [Path('1-A.json')]}
        positions = np.random.randn(20, 3)
        timestamps = np.linspace(0, 10, 20)
        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(positions, None, None, timestamps)):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1'], [])
        assert 'Person_1' in result
        assert '1-A' in result['Person_1']

    def test_axes_reordered_xzy(self):
        mock_group = {'Person_1': [Path('1-A.json')]}
        positions = np.arange(60).reshape(20, 3).astype(float)
        timestamps = np.linspace(0, 10, 20)
        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(positions, None, None, timestamps)):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1'], [])
        loaded = result['Person_1']['1-A']
        np.testing.assert_array_equal(loaded[:, 0], positions[:, 0])
        np.testing.assert_array_equal(loaded[:, 1], positions[:, 2])
        np.testing.assert_array_equal(loaded[:, 2], positions[:, 1])

    def test_multiple_persons_loaded(self):
        mock_group = {
            'Person_1': [Path('1-A.json')],
            'Person_2': [Path('2-A.json')],
        }
        positions = np.random.randn(20, 3)
        timestamps = np.linspace(0, 10, 20)
        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(positions, None, None, timestamps)):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1', 'Person_2'], [])
        assert 'Person_1' in result
        assert 'Person_2' in result

    def test_person_with_no_valid_scenarios_excluded(self):
        mock_group = {'Person_1': [Path('1-A.json')]}
        padded = np.zeros((20, 3))
        timestamps = np.linspace(0, 10, 20)
        with patch('scenario_comparison.group_files_by_person', return_value=mock_group), \
             patch('scenario_comparison.load_head_data',
                   return_value=(padded, None, None, timestamps)):
            result = scenario_comparison.load_scenario_positions_for_persons(['Person_1'], [])
        assert 'Person_1' not in result