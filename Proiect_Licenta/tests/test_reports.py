import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

for mod in [
    'core', 'core.processing', 'core.processing.data_loader',
    'core.processing.feature_extractor',
    'core.analysis', 'core.analysis.disability_engine',
    'utils', 'utils.file_management',
    'mpl_toolkits', 'mpl_toolkits.mplot3d',
    'reportlab', 'reportlab.platypus', 'reportlab.lib',
    'reportlab.lib.styles', 'reportlab.lib.colors',
    'reportlab.lib.units',
]:
    sys.modules[mod] = MagicMock()

mock_config = MagicMock()
mock_config.USE_DATA_DRIVEN_THRESHOLDS = False
mock_config.DISABILITY_SCORE_THRESHOLDS = {'HIGH': 0.72, 'MEDIUM': 0.61, 'LOW': 0.52}
mock_config.DATA_DRIVEN_THRESHOLD_QUANTILES = {'LOW': 0.25, 'MEDIUM': 0.5, 'HIGH': 0.75}

sys.modules['config'] = MagicMock()
sys.modules['config'].disability_config = mock_config

from src.visualization import reports



def make_assessment(n=4, statuses=None):
    if statuses is None:
        statuses = ['HIGH', 'MEDIUM', 'LOW', 'NONE']
    assessment = {}
    for i in range(n):
        status = statuses[i % len(statuses)]
        assessment[f'Person_{i+1}'] = {
            'final_score': 0.8 - i * 0.1,
            'status': status,
            'cluster': i % 2,
            'distance_score': 0.5 + i * 0.05,
            'mahalanobis_score': 0.4 + i * 0.03,
            'consistency_score': 0.3 + i * 0.02,
        }
    return assessment


def make_features(n=4, n_feat=15, seed=0):
    np.random.seed(seed)
    return {f'Person_{i+1}': np.random.randn(n_feat) for i in range(n)}


class TestCreateVisualDisabilityReport:

    def test_empty_assessment_returns_early(self, capsys):
        with patch('reports.config', mock_config):
            reports.create_visual_disability_report({})
        assert 'No data' in capsys.readouterr().out

    def test_runs_without_exception(self, tmp_path):
        assessment = make_assessment()
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_visual_disability_report(assessment)

    def test_nan_scores_excluded(self, tmp_path, capsys):
        assessment = {
            'Person_1': {'final_score': float('nan'), 'status': 'NONE'},
            'Person_2': {'final_score': 0.5, 'status': 'LOW'},
        }
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_visual_disability_report(assessment)

    def test_all_nan_scores_returns_early(self, tmp_path, capsys):
        assessment = {
            'Person_1': {'final_score': float('nan'), 'status': 'NONE'},
        }
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_visual_disability_report(assessment)
        assert 'No valid' in capsys.readouterr().out

    def test_all_statuses_handled(self, tmp_path):
        assessment = make_assessment(n=4, statuses=['HIGH', 'MEDIUM', 'LOW', 'NONE'])
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_visual_disability_report(assessment)

    def test_single_person(self, tmp_path):
        assessment = {'Person_1': {'final_score': 0.6, 'status': 'MEDIUM'}}
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_visual_disability_report(assessment)

    def test_missing_final_score_key_handled(self, tmp_path):
        assessment = {
            'Person_1': {'status': 'NONE'},
            'Person_2': {'final_score': 0.5, 'status': 'LOW'},
        }
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_visual_disability_report(assessment)


class TestPlotPersonDendrogram:

    def test_empty_person_ids_returns_none(self, tmp_path, capsys):
        features = make_features()
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, [])
        assert result is None

    def test_single_person_returns_features_key(self, tmp_path):
        features = make_features(n=1)
        person_ids = ['Person_1']
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, person_ids)
        assert result is not None
        assert 'features' in result
        assert result['scores'] is None

    def test_multiple_persons_returns_dict(self, tmp_path):
        features = make_features(n=4)
        person_ids = list(features.keys())
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, person_ids)
        assert result is not None
        assert 'features' in result
        assert 'scores' in result

    def test_with_disability_assessment_generates_scores_dendrogram(self, tmp_path):
        features = make_features(n=4)
        person_ids = list(features.keys())
        assessment = make_assessment(n=4)
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, person_ids, disability_assessment=assessment)
        assert result is not None
        assert result['scores'] is not None

    def test_without_assessment_scores_path_is_none(self, tmp_path):
        features = make_features(n=4)
        person_ids = list(features.keys())
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, person_ids, disability_assessment=None)
        assert result['scores'] is None

    def test_partial_assessment_skips_scores_dendrogram(self, tmp_path):
        features = make_features(n=4)
        person_ids = list(features.keys())
        partial_assessment = {'Person_1': make_assessment(n=1)['Person_1']}
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, person_ids, disability_assessment=partial_assessment)
        assert result['scores'] is None

    def test_two_persons_minimum_dendrogram(self, tmp_path):
        features = make_features(n=2)
        person_ids = list(features.keys())
        with patch('reports.os.path.dirname', return_value=str(tmp_path)):
            result = reports.plot_person_dendrogram(features, person_ids)
        assert result is not None


class TestCreateDisabilityReport:

    def test_empty_assessment_returns_none(self, capsys):
        result = reports.create_disability_report({})
        assert result is None
        assert 'No data' in capsys.readouterr().out

    def test_returns_dict_with_txt_and_pdf_keys(self, tmp_path):
        assessment = make_assessment()
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            result = reports.create_disability_report(assessment)
        assert result is not None
        assert 'txt' in result
        assert 'pdf' in result

    def test_all_statuses_appear_in_report(self, tmp_path, capsys):
        assessment = make_assessment(n=4, statuses=['HIGH', 'MEDIUM', 'LOW', 'NONE'])
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_disability_report(assessment)
        out = capsys.readouterr().out
        for status in ['high', 'medium', 'low', 'none']:
            assert status in out.lower()

    def test_single_person_report(self, tmp_path):
        assessment = {'Person_1': {
            'final_score': 0.3,
            'status': 'NONE',
            'cluster': 0,
            'distance_score': 0.2,
            'mahalanobis_score': 0.1,
            'consistency_score': 0.4,
        }}
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            result = reports.create_disability_report(assessment)
        assert result is not None

    def test_report_contains_total_persons(self, tmp_path, capsys):
        assessment = make_assessment(n=3)
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            reports.create_disability_report(assessment)
        out = capsys.readouterr().out
        assert '3' in out

    def test_does_not_raise_with_all_same_status(self, tmp_path):
        assessment = make_assessment(n=3, statuses=['NONE'])
        with patch('reports.os.path.dirname', return_value=str(tmp_path)), \
             patch('reports.config', mock_config):
            result = reports.create_disability_report(assessment)
        assert result is not None