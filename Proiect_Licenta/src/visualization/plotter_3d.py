from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from scipy.cluster.hierarchy import linkage, dendrogram
import csv
import json
import os
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from src.core.processing.data_loader import load_head_data
from src.core.processing.feature_extractor import extract_behavior_features
from src.core.analysis.clustering import perform_agglomerative_clustering
from src.core.analysis.disability_engine import (
    analyze_person_disability,
    detect_disability_patterns_unsupervised,
    save_detailed_results
)
from src.utils.file_management import group_files_by_person
from src.config import disability_config as config
from src.visualization.reports import (
    create_visual_disability_report,
    plot_person_dendrogram,
    create_disability_report
)
from src.visualization.scenario_comparison import (
    plot_all_scenarios_comparison,
    plot_person_scenario_comparison
)
    
def plot_clusters(positions, labels, scenario_name, combo_name):
    """
    Visualizes and summarizes the spatial clusters obtained from 3D positional
    data.

    The function generates a three-dimensional scatter plot in which points are
    colored according to their assigned cluster. In addition, it computes
    descriptive statistics for each cluster, including its centroid, size, and
    relative proportion of the total observations. The visualization is saved as
    an image, while the computed cluster statistics are exported as a JSON file
    for further analysis.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape ``(N, 3)`` containing the three-dimensional coordinates
        of the recorded positions.

    labels : numpy.ndarray
        One-dimensional array containing the cluster label assigned to each
        position by the clustering algorithm.

    scenario_name : str
        Name of the analyzed scenario. This value is used in the output
        filenames and metadata.

    combo_name : str
        Name of the feature or sensor combination used during clustering.
        This value is included in the output filenames and metadata.

    Returns
    -------
    list[dict]
        A list containing descriptive statistics for each detected cluster.
        Each dictionary includes:

        - ``label`` : int
            Cluster identifier.
        - ``color`` : str
            Color assigned to the cluster in the visualization.
        - ``center`` : dict
            Centroid coordinates with keys ``x``, ``y``, and ``z``.
        - ``size`` : int
            Number of samples assigned to the cluster.
        - ``percentage`` : float
            Percentage of the total samples belonging to the cluster.

    Outputs
    -------
    The function generates the following files in the output directory:

    - ``cluster_<scenario>_<combo>.png``:
    Three-dimensional visualization of the clustered positions.
    - ``clusters_<scenario>_<combo>.json``:
    JSON file containing the computed cluster statistics and metadata.

    Notes
    -----
    - Cluster centroids represent the mean spatial location of each cluster.
    - Cluster size and percentage provide an estimate of the relative amount of
    time spent within each spatial region.
    - Visualization colors are assigned from a predefined color palette and are
    reused if the number of clusters exceeds the number of available colors.
    - The viewing angle and axis orientation are configured to match the
    coordinate conventions used in the virtual reality environment.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    cluster_colors = getattr(config, 'CLUSTER_COLORS', ['red', 'blue', 'green', 'orange', 'purple'])

    if len(unique_labels) > len(cluster_colors):
        print(f"Warning: {len(unique_labels)} clusters but only {len(cluster_colors)} colors available.")
    
    colors = [cluster_colors[i % len(cluster_colors)] for i in range(len(unique_labels))]

    cluster_stats = []

    for label, color in zip(unique_labels, colors):
        idx = labels == label
        cluster_positions = positions[idx]

        center = np.mean(cluster_positions, axis=0)
        size = len(cluster_positions)
        percentage = (size / len(positions)) * 100

        cluster_stats.append({
            'label': int(label),
            'color': color,
            'center': {
                'x': float(center[0]),
                'y': float(center[1]),
                'z': float(center[2])
            },
            'size': size,
            'percentage': percentage
        })

        ax.scatter(
            cluster_positions[:, 0], 
            cluster_positions[:, 1], 
            cluster_positions[:, 2],
            c=[color], 
            label=f'Area {label} ({size} points, {percentage:.1f}%)',
            s=50, 
            alpha=0.7
        )

        ax.scatter(center[0], center[1], center[2],
                   c='black', s=200, marker='x', linewidth=3, zorder=10)
        
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position', fontsize=12, fontweight='bold')
    # ax.set_title(f'Head Position Grouping - {scenario_name}\n(Areas where user spent most time)',
    #              fontsize=14, fontweight='bold')
    
    # ax.legend(fontsize=11)
    ax.grid(True, alpha=0.5)
    ax.view_init(elev=30, azim=120)

    safe_combo = combo_name.replace("+", "-")
    image_path = os.path.join(output_dir, f'cluster_{scenario_name}_{safe_combo}.png')

    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()

    stats_path = os.path.join(
        output_dir, f'clusters_{scenario_name}_{safe_combo}.json'
    )

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'scenario': scenario_name,
            'combo': combo_name,
            'total_points': len(positions),
            'clusters': cluster_stats
        }, f, indent=2)

    print(f"\nCluster plot saved to {image_path}")
    print(f"Cluster stats saved to {stats_path}")

    return cluster_stats


def plot_disability_annotations(all_data, disability_likelihood, global_origin=None):
    """
    Generates three-dimensional trajectory visualizations annotated according to
    the estimated disability likelihood for each analyzed scenario.

    For each scenario, the function extracts the recorded positional data,
    normalizes the trajectory relative to a common or local spatial origin, and
    renders a 3D path colored according to the corresponding disability
    likelihood category. Each visualization is saved as an individual
    high-resolution image to facilitate qualitative analysis and reporting.

    Parameters
    ----------
    all_data : dict
        Dictionary mapping scenario names to their corresponding data objects.
        Each data object must contain a ``"data"`` field whose first three
        columns represent the three-dimensional spatial coordinates
        (x, y, z).

    disability_likelihood : dict
        Dictionary mapping scenario names to their estimated disability
        likelihood. Each entry is expected to contain:

        - ``status`` : str
            Disability likelihood category (e.g., ``"HIGH"``, ``"MEDIUM"``,
            ``"LOW"``, or ``"NONE"``).
        - ``score`` : float
            Normalized behavioral variation score in the range [0, 1].

    global_origin : numpy.ndarray, optional
        Three-dimensional reference point used to normalize all trajectories to
        a common coordinate system. If ``None``, each trajectory is normalized
        relative to its initial position.

    Outputs
    -------
    For each analyzed scenario, the function generates:

    - ``<scenario_name>_3d_path.png``

    stored in:

    ``data/output/disability_3d/<scenario_name>/``

    Notes
    -----
    - Trajectories are normalized before visualization to improve
    comparability across scenarios.
    - Trajectory color represents the estimated disability likelihood.
    - The initial position of each trajectory is highlighted to indicate the
    starting location.
    - Scenarios without an associated disability likelihood score are skipped.
    - Each scenario is visualized independently, producing one figure per
    trajectory.
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base_dir = os.path.join(base_dir, "data", "output", "disability_3d")
    os.makedirs(output_base_dir, exist_ok=True)

    

    colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green', 'NONE': 'blue'}


    for scenario_name, data in all_data.items():

        if scenario_name not in disability_likelihood:
            print(f"Skippimg {scenario_name} as it has no disability likelihood score")
            continue


        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        positions = data["data"][:, :3]

        if global_origin is not None:
            positions = positions - global_origin
        else:
            positions = positions - positions[0]

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        ax.invert_xaxis()
        ax.invert_yaxis()

        status = disability_likelihood[scenario_name]['status']
        score = disability_likelihood[scenario_name]['score']

        ax.plot(x, y, z, color=colors[status], alpha=0.85, linewidth=2.5)
        ax.scatter(x[0], y[0], z[0], color=colors[status], s=120, marker='o', edgecolors='black')
        
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z Position', fontsize=12, fontweight='bold')
        ax.set_zlabel('Y Position', fontsize=12, fontweight='bold')
        # ax.set_title('3D Head Paths - Disability Likelihood', fontsize=16, fontweight='bold')

        # legend_elements = [
        #     Line2D([0], [0], color='red', lw=2, label='High Disability Likelihood'),
        #     Line2D([0], [0], color='orange', lw=2, label='Medium Disability Likelihood'),
        #     Line2D([0], [0], color='green', lw=2, label='Low Disability Likelihood')
        # ]
        # ax.legend(handles=legend_elements, fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))

        # explanation = (
        #     "Behavioral Variation Analysis:\n"
        #     "RED = Significant behavioral variation\n"
        #     "ORANGE = Moderate behavioral variation\n"
        #     "GREEN = Typical behavioral variation\n"
        # )

        # plt.figtext(0.02, 0.02, explanation, fontsize=9, bbox=dict(facecolor='lightblue', alpha=0.5))

        scenario_dir = os.path.join(output_base_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        file_path = os.path.join(scenario_dir, f"{scenario_name}_3d_path.png")
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved 3D plot for {scenario_name} at: {file_path}")

    print("\nAll 3D plots saved sucessfully")


# if __name__ == "__main__":
#     print("\nDebug: Running plotter_3d.py workflow from main.py")

#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     vr_dir = Path(base_dir) / "data" / "vr_recordings"

#     json_files = sorted(vr_dir.glob("*.json"))
#     grouped = group_files_by_person(json_files)
#     selected_persons = sorted(grouped.keys())[:3]
#     grouped = {person: grouped[person] for person in selected_persons}

#     print(f"Total JSON files: {len(json_files)}")
#     print(f"Total persons found: {len(grouped)} (first 3 selected: {selected_persons})")

#     person_aggregated_data = {}
#     all_valid_scenarios_data = {}
#     hierarchical_data = {}

#     for person, files in grouped.items():
#         print(f"\nProcessing {person}")
#         person_features = []
#         total_scenarios = len(files)
#         hierarchical_data[person] = {}

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
#             scenario_name = f"{person}_{file_path.stem}"
#             all_valid_scenarios_data[scenario_name] = {
#                 "data": scenario_data,
#                 "person": person
#             }
#             hierarchical_data[person][file_path.stem] = scenario_data

#             if len(positions) > 10:
#                 kmeans = KMeans(n_clusters=3, n_init=10)
#                 labels = kmeans.fit_predict(positions)
#                 plot_clusters(positions, labels, scenario_name, "Head_Position")

#         if hierarchical_data[person]:
#             print(f"Generate comparison between scenarios for {person}")
#             plot_person_scenario_comparison(hierarchical_data[person], person)

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

#     if hierarchical_data:
#         print("\nGenerate global comparison between persons")
#         plot_all_scenarios_comparison(hierarchical_data)

#     if person_aggregated_data:
#         person_ids = list(person_aggregated_data.keys())
#         features_matrix = np.array(list(person_aggregated_data.values()))

#         print(f"Clustering on {len(person_ids)} persons with feature matrix shape: {features_matrix.shape}")
#         cluster_labels = perform_agglomerative_clustering(
#             data=features_matrix,
#             scenario_name="Across_Scenarios_Analysis",
#             features_combination="Aggregated_Behavior_Features"
#         )

#         if cluster_labels is not None:
#             print("\nClustering results:")
#             results_summary = {}
#             for idx, label in enumerate(cluster_labels):
#                 person = person_ids[idx]
#                 print(f" Person: {person}, Cluster: {label}")
#                 results_summary.setdefault(label, []).append(person)

#             for cluster_id, members in results_summary.items():
#                 print(f" Cluster {cluster_id}: {len(members)} members - {members}")
#         else:
#             print("\nError: Not enough data to perform clustering.")

#         disability_assessment = analyze_person_disability(person_aggregated_data)

#         print("\nGenerate report for behavioral assessment:")
#         create_visual_disability_report(disability_assessment)

#         print("\nGenerate dendrogram for behavioral similarity:")
#         plot_person_dendrogram(person_aggregated_data, person_ids)

#         print("\nGenerate .txt and .pdf reports")
#         report_paths = create_disability_report(disability_assessment)
#         if report_paths:
#             print(f"Reports generated successfully: {report_paths}")

#         formatted_likelihood = {}
#         for scenario_name, obj in all_valid_scenarios_data.items():
#             p_id = obj["person"]
#             if p_id in disability_assessment:
#                 formatted_likelihood[scenario_name] = {
#                     'status': disability_assessment[p_id]['status'],
#                     'score': disability_assessment[p_id]['final_score'],
#                 }

#         print("formatted_likelihood size:", len(formatted_likelihood))

#         if formatted_likelihood:
#             print("\nGenerating disability annotation plots...")
            
#             plot_disability_annotations(all_valid_scenarios_data, formatted_likelihood)

#         if disability_assessment:
#             print("\nBehavioral assessment results:")
#             for p_id, result in disability_assessment.items():
#                 print(f" Person: {p_id}, Status: {result['status']}, Final Score: {result['final_score']}, Cluster(K-means): {result['cluster']}")

#             save_detailed_results(disability_assessment, person_aggregated_data)
#     else:
#         print("No valid data available for clustering and disability analysis.")