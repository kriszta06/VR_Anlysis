import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import os
from pathlib import Path
from core.processing.data_loader import load_head_data
from core.processing.feature_extractor import extract_behavior_features
from utils.file_management import group_files_by_person
from config import disability_config as config


def perform_agglomerative_clustering(data, scenario_name, features_combination):
    '''
    Performs hierarchical agglomerative clustering on scenario data,
    generates a dendrogram visualzation, and returns cluster labels.

    The function standardizes the input data, computes hierarchical clustering using Ward linkage, saves a dendrogram image, and 
    assigns each sample to one of three clusters.

    Parameters:
    
    :param data: numpy.ndarray
    2D array of shape (n_samples, n_features).
    Each row represents a data point and each column represents
    a behavioral feature.

    :param scenario_name: str
    Name of the scenario being analyzed.
    Used for debug messages and output file naming.

    :param features_combination: str
    Description of the feature set used.
    Used for labeling and output naming.
    If it contains "Position", spatial cluster interpretation is performed.

    Returns:

    numpy.ndarray or None
    1D array of cluster labels (length = n_samples),
    where each value represents the assigned cluster.
    Returns None if no valid data is provided.

    Notes:
    - Data is standardized using StandardScaler befor clustering.
    - Ward linkage is used to create compact, variance-minimizing clusters.
    - A dendrogram visualization is saved to disk.
    - The number of clusters is fixed at 3. (low/medium/high)

    '''
    if len(data) == 0:
        print(f"No valid data for scenario '{scenario_name}'")
        return None

    if len(data) == 1:
        print(f"Only one sample available for scenario '{scenario_name}'. Assigning single cluster label.")
        return np.array([0])

    print(f"Performing clustering for scenario '{scenario_name}' with {len(data)} samples")
    print(f"Features combination: {features_combination}")
    print(f"Total number of points: {len(data)}")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    z = linkage(scaled_data, method='ward')
    plt.figure(figsize=(15, 10))

    node_label = [str(i+1) for i in range(len(data))]

    dendrogram(
        z, 
        orientation='right',
        labels=node_label,
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_font_size=10,
        leaf_rotation=0
        )
    
    # plt.title(f'Dendrogram for {scenario_name}\n (Shows how similar points are grouped)', fontsize=14, fontweight='bold')
    plt.xlabel('Point index', fontsize=12)
    plt.ylabel('Distance between groups', fontsize=12)

    # plt.figtext(0.02, 0.02,
    #             "DENDROGRAM EXPLANATION:\n"
    #             "Horizontal lines join subjects with similar behavior.\n"
    #             "The closer the vertical split is to the left, \n"
    #             "the more similar the subjects are.\n"
    #             "X-Axis (Distance) represents the degree of\n"
    #             "behavioral deviation.\n"
    #             "Large group (long lines) = major differences between in motor patterns",
    #             fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.8))
    
    plt.savefig(f'./data/output/dendrogram/dendrogram_{scenario_name}_{features_combination.replace("+", "_")}.png',
                dpi=300, 
                bbox_inches='tight'
    )

    plt.close()

    n_clusters = min(3, len(data))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(scaled_data)

    print(f"\nClustering results for scenario '{scenario_name}':")
    print(f"Number of grous created: {n_clusters}")

    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        percentage = (cluster_size / len(cluster_labels)) * 100
        print(f" Group {i}: {cluster_size} points ({percentage:.2f}%)")

    if 'Position' in features_combination:
        print(f"\nCluster interpretation for scenario '{scenario_name}':")
        positions = data[:, :3] if data.shape[1] >= 3 else data
        for i in range(n_clusters):
            cluster_positions = positions[cluster_labels == i]
            if len(cluster_positions) > 0:
                center =np.mean(cluster_positions, axis=0)
                print(f" Group {i} center position: {center}")
                if len(cluster_positions) > 1:
                    spread = np.std(cluster_positions, axis=0)
                    print(f" Group {i} position spread: {spread}")
            
    return cluster_labels


# if __name__ == "__main__": 

#     # print("DEBUG : Running clustering script")

#     # np.random.seed(42)

 
#     # os.makedirs("./data/output/dendrogram", exist_ok=True)
#     # os.makedirs("./data/output/clustering", exist_ok=True)

#     # cluster_1 = np.random.normal(loc=[0, 0, 0], scale=0.5, size=(30, 3))
#     # cluster_2 = np.random.normal(loc=[5, 5, 5], scale=0.5, size=(30, 3))
#     # cluster_3 = np.random.normal(loc=[10, 0, 5], scale=0.5, size=(30, 3))

#     # data = np.vstack([cluster_1, cluster_2, cluster_3])

#     # scenario_name = "Test_Scenario"
#     # features_combination = "Position"

#     # print(f"Debug: Data shape: {data.shape}")
#     # print(f"Debug: Scenario name: {scenario_name}")
#     # print(f"Debug: Features combination: {features_combination}")

#     # labels = perform_agglomerative_clustering(data, scenario_name, features_combination)

#     # if labels is None:
#     #     print(f"Debug: Clustering failed")
#     # else:
#     #     print(f"Debug: Clustering finished successfully")
#     #     print(f"Debug: labels shape: {labels.shape}")
#     #     print(f"Debug: Unique labels: {np.unique(labels)}")

#     # print("DEBUG : Clustering script finished")


#     print("DEBUG: Running clustering on first 3 VR JSON files")

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
#     vr_dir = Path(PROJECT_ROOT) / "data" / "vr_recordings"

#     json_files = sorted(vr_dir.glob("*.json"))
#     grouped = group_files_by_person(json_files)
#     selected_persons = sorted(grouped.keys())[:3]

#     print(f"Total JSON files: {len(json_files)}")
#     print(f"Total persons: {len(grouped)}")
#     print(f"Selected persons: {selected_persons}")

#     person_aggregated_data = {}
#     os.makedirs(os.path.join(PROJECT_ROOT, "data", "output", "dendrogram"), exist_ok=True)

#     for person in selected_persons:
#         files = grouped.get(person, [])
#         print(f"\nProcessing {person}")
#         person_features = []
#         total_scenarios = len(files)

#         print(f"Total scenarios for person {person}: {total_scenarios}")

#         for idx, file_path in enumerate(files):
#             print(f"\n Scenario {idx+1}/{total_scenarios}: {file_path.name}")
#             positions, rotations, forward_vectors, timestamps = load_head_data(file_path)

#             print(
#                 f"Loaded timestamps: {len(timestamps)}, "
#                 f"positions: {positions.shape}, "
#                 f"rotations: {rotations.shape}, "
#                 f"forward_vectors: {forward_vectors.shape}"
#             )

#             if len(timestamps) == 0:
#                 print(f"Skipping {file_path.name} due to no valid records.")
#                 continue

#             if positions.shape[1] >= 3:
#                 positions = positions[:, [0, 2, 1]]
#                 rotations = rotations[:, [0, 2, 1]]
#                 forward_vectors = forward_vectors[:, [0, 2, 1]]
#                 print("Axes reordered to [X,Z,Y] for VR consistency.")

#             padded_ratio = np.all(positions == 0, axis=1).mean()
#             if padded_ratio > 0.5:
#                 print(f"Warning: {file_path.name} has incomplete data (padded). File will be ignored")
#                 continue

#             print("Extracting features")
#             scenario_data = np.hstack((positions, rotations, forward_vectors, timestamps.reshape(-1, 1)))
#             features = extract_behavior_features(scenario_data)

#             print("Features extracted:")
#             print(features)
#             print(f"Feature vector length: {len(features)}")

#             person_features.append(features)

#         actual_valid_scenarios = len(person_features)
#         corruption_rate = (total_scenarios - actual_valid_scenarios) / total_scenarios if total_scenarios > 0 else 1

#         print(f"Person: {person}")
#         print(f" Total scenarios: {total_scenarios}")
#         print(f" Valid scenarios: {actual_valid_scenarios}")
#         print(f" Corruption rate: {corruption_rate:.2f}")
#         print(f"MIN_SCENARIOS_REQUIRED: {config.MIN_SCENARIOS_REQUIRED}")
#         print(f"MAX_CORRUPTION_RATE: {config.MAX_CORRUPTION_RATE}")

#         if actual_valid_scenarios >= config.MIN_SCENARIOS_REQUIRED and corruption_rate <= config.MAX_CORRUPTION_RATE:
#             person_aggregated_data[person] = np.mean(person_features, axis=0)
#             print(f"{person} is selected for analysis")
#             print(person_aggregated_data[person])
#         else:
#             print(f"{person} is excluded from analysis due to insufficient valid scenarios or high corruption rate")

#     if len(person_aggregated_data) < 2:
#         print("\nNot enough persons for clustering. Need at least 2 aggregated persons.")
#     else:
#         features_matrix = np.array(list(person_aggregated_data.values()))
#         person_ids = list(person_aggregated_data.keys())
#         print(f"\nClustering {len(person_ids)} persons with feature matrix shape: {features_matrix.shape}")

#         labels = perform_agglomerative_clustering(
#             data=features_matrix,
#             scenario_name="First_3_Persons",
#             features_combination="Aggregated_Behavior_Features"
#         )

#         if labels is not None:
#             print("\nClustering results:")
#             for idx, person in enumerate(person_ids):
#                 print(f" Person: {person}, Cluster: {labels[idx]}")
#         else:
#             print("Clustering did not return labels.")

