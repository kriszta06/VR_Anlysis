import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from src.core.analysis import clustering



def make_blob_data(seed=42):
    rng = np.random.default_rng(seed)
    cluster_1 = rng.normal(loc=[0, 0, 0], scale=0.3, size=(10, 3))
    cluster_2 = rng.normal(loc=[10, 10, 10], scale=0.3, size=(10, 3))
    return np.vstack([cluster_1, cluster_2])


class TestPerformAgglomerativeClustering:

    def test_empty_data_returns_none(self, capsys):
        data = np.empty((0, 3))
        result = clustering.perform_agglomerative_clustering(data, "Scenario_A", "Position")
        assert result is None
        captured = capsys.readouterr()
        assert "No valid data" in captured.out

    def test_single_sample_returns_single_cluster_label(self, capsys):
        data = np.array([[1.0, 2.0, 3.0]])
        result = clustering.perform_agglomerative_clustering(data, "Scenario_A", "Position")
        np.testing.assert_array_equal(result, np.array([0]))
        captured = capsys.readouterr()
        assert "Only one sample available" in captured.out

    def test_multiple_samples_returns_labels_with_correct_length(self, monkeypatch, tmp_path):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        result = clustering.perform_agglomerative_clustering(data, "Scenario_A", "Behavior_Features")
        assert result is not None
        assert len(result) == len(data)

    def test_two_blobs_separate_into_two_clusters(self, monkeypatch, tmp_path):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        result = clustering.perform_agglomerative_clustering(data, "Scenario_A", "Behavior_Features")
        assert len(np.unique(result)) == 2
        assert len(set(result[:10])) == 1
        assert len(set(result[10:])) == 1
        assert result[0] != result[10]

    def test_n_clusters_capped_by_sample_count(self, monkeypatch, tmp_path):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 10, raising=False)

        data = make_blob_data(seed=1)[:3]
        result = clustering.perform_agglomerative_clustering(data, "Scenario_A", "Behavior_Features")
        assert len(np.unique(result)) <= 3

    def test_dendrogram_file_is_created(self, monkeypatch, tmp_path):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        clustering.perform_agglomerative_clustering(data, "Scenario_A", "Behavior_Features")
        saved_files = list(tmp_path.glob("dendrogram_Scenario_A_*.png"))
        assert len(saved_files) == 1

    def test_dendrogram_filename_replaces_plus_signs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        clustering.perform_agglomerative_clustering(data, "Scenario_A", "Position+Rotation")
        expected_path = tmp_path / "dendrogram_Scenario_A_Position_Rotation.png"
        assert expected_path.exists()

    def test_position_features_prints_cluster_centers(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        clustering.perform_agglomerative_clustering(data, "Scenario_A", "Position")
        captured = capsys.readouterr()
        assert "Cluster interpretation" in captured.out
        assert "center position" in captured.out

    def test_non_position_features_skip_center_interpretation(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        clustering.perform_agglomerative_clustering(data, "Scenario_A", "Behavior_Features")
        captured = capsys.readouterr()
        assert "Cluster interpretation" not in captured.out

    def test_accepts_matching_person_ids_without_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(clustering.config, "DENDROGRAM_OUTPUT_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(clustering.config, "AGGLOMERATIVE_N_CLUSTERS", 2, raising=False)

        data = make_blob_data()
        person_ids = [f"Person_{i}" for i in range(len(data))]
        result = clustering.perform_agglomerative_clustering(data, "Scenario_A", "Behavior_Features", person_ids=person_ids)
        assert result is not None
        assert len(result) == len(data)