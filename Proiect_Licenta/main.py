from pathlib import Path
import os
import numpy as np
import csv
import json
import seaborn as sns
from sklearn.cluster import KMeans
from core.processing.data_loader import load_head_data
from core.processing.feature_extractor import extract_behavior_features
from core.analysis.clustering import perform_agglomerative_clustering
from core.analysis.disability_engine import analyze_person_disability, detect_disability_patterns_unsupervised, save_detailed_results
from utils.file_management import group_files_by_person
from config import disability_config as config
from visualization.plotter_3d import plot_clusters, plot_disability_annotations
from visualization.reports import create_visual_disability_report, plot_person_dendrogram, create_disability_report 
from visualization.scenario_comparison import plot_all_scenarios_comparison, plot_person_scenario_comparison
from evaluation.ground_truth_handler import load_ground_truth, sync_data
from evaluation.metrics import get_output_path, calculate_and_plot_metrics, export_evaluation_summary

os.makedirs("data/output/dendrogram", exist_ok=True)
# os.makedirs("data/output/clustering", exist_ok=True)
# os.makedirs("data/output/disability_results", exist_ok=True)
# os.makedirs("data/output/plots", exist_ok=True)
# os.makedirs("data/output/disability_3d", exist_ok=True)

json_files = list(Path("data/vr_recordings").glob("*.json"))
grouped = group_files_by_person(json_files)

person_aggregated_data = {}
all_valid_scenarios_data = {}
hierarchical_data = {}

print(f"Total JSON files: {len(json_files)}\n")
print(f"Total persons: {len(grouped)}\n")

for person, files in grouped.items():
    print(f"\nProcessing {person}")
    person_features = []
    total_scenarios = len(files)
    hierarchical_data[person] = {}

    print(f"Total scenarios for person {person}: {total_scenarios}")

    for idx, file_path in enumerate(files):

        print(f"\n Scenario {idx+1}/{total_scenarios}: {file_path.name}")

        positions, rotations, forward_vectors, timestamps = load_head_data(file_path)

        print(f"Loaded timestamps: {len(timestamps)}, positions: {positions.shape}, rotations: {rotations.shape}, forward_vectors: {forward_vectors.shape}")

        if len(timestamps) == 0:
            print(f"Skipping {file_path.name} due to no valid records.")
            continue

        if positions.shape[1] >= 3:
            positions = positions[:, [0,2,1]]
            rotations = rotations[:, [0,2,1]]
            forward_vectors = forward_vectors[:, [0,2,1]]
            print("Axes reordered to [X,Z,Y] for VR consistency.")
            
        padded_ratio = np.all(positions == 0, axis=1).mean()
        is_padded = padded_ratio > 0.5

        if is_padded:
            print(f"Warning: {file_path.name} has incomplete data (padded). File will be ignored")
            continue

        print("Extracting features")
        scenario_data = np.hstack((positions, rotations, forward_vectors, timestamps.reshape(-1, 1)))
        features = extract_behavior_features(scenario_data)

        print("Features extracted:")
        print(features)
        print(f"Feature vector length: {len(features)}")
        
        person_features.append(features)
        print(f"Valid scenarios processed for {person}: {len(person_features)} out of {total_scenarios}")

        scenario_name = f"{person}_{file_path.stem}"
        all_valid_scenarios_data[scenario_name] = {
            "data": scenario_data,
            "person": person
        }

        scenario_type = file_path.stem
        hierarchical_data[person][scenario_type] = scenario_data


        if len(positions) > 10:
            kmeans = KMeans(n_clusters=3, n_init=10)
            labels = kmeans.fit_predict(positions)
            plot_clusters(positions, labels, scenario_name, "Head_Position")

    if hierarchical_data[person]:
        print(f"Generate comparison between scenarios for {person}")
        plot_person_scenario_comparison(hierarchical_data[person], person)

    actual_valid_scenarios = len(person_features)
    corruption_rate = (total_scenarios - actual_valid_scenarios) / total_scenarios if total_scenarios > 0 else 1
    
    print(f"Person: {person}")
    print(f" Total scenarios: {total_scenarios}")
    print(f" Valid scenarios: {actual_valid_scenarios}")
    print(f" Corruption rate: {corruption_rate:.2f}")
    print(f"MIN_SCENARIOS_REQUIRED: {config.MIN_SCENARIOS_REQUIRED}")
    print(f"MAX_CORRUPTION_RATE: {config.MAX_CORRUPTION_RATE}")

    if actual_valid_scenarios >= config.MIN_SCENARIOS_REQUIRED and corruption_rate <= config.MAX_CORRUPTION_RATE:
        person_mean = np.mean(person_features, axis=0)
        person_std = np.std(person_features, axis=0) if len(person_features) > 1 else np.zeros_like(person_mean)
        person_aggregated_data[person] = np.concatenate((person_mean, person_std))
        print(f"{person} is selected for analysis")
        print("Aggregated features for person:")
        print(person_aggregated_data[person])
    else:
        print(f"{person} is excluded from analysis due to insufficient valid scenarios or high corruption rate")

if hierarchical_data:
    print("\nGenerate global comparison betweem persons")
    plot_all_scenarios_comparison(hierarchical_data)

print(f"Total selected persons: {len(person_aggregated_data)}")

if person_aggregated_data:
    person_ids = list(person_aggregated_data.keys())
    features_matrix = np.array(list(person_aggregated_data.values()))

    print(f"Clustering on {len(person_ids)} persons with feature matrix shape: {features_matrix.shape}")

    cluster_labels = perform_agglomerative_clustering(
        data = features_matrix,
        scenario_name="Across_Scenarios_Analysis",
        features_combination="Aggregated_Behavior_Features"
    )

    if cluster_labels is not None:
        print("\nClustering results:")
        results_summary = {}
        for idx, label in enumerate(cluster_labels):
            person = person_ids[idx]
            print(f" Person: {person}, Cluster: {label}")

            if label not in results_summary:
                results_summary[label] = []
            results_summary[label].append(person)

        for cluster_id, members in results_summary.items():
            print(f" Cluster {cluster_id}: {len(members)} members - {members}")
    else:
        print("\nError: Does not have enough data to perform clustering.")

    disability_assessment = analyze_person_disability(person_aggregated_data)

    print("\nGenerate report for behavioral assessment:")
    create_visual_disability_report(disability_assessment)

    print("\nGenerate dendrogram for behavioral similarity:")
    person_ids = list(person_aggregated_data.keys())
    plot_person_dendrogram(person_aggregated_data, person_ids, disability_assessment=disability_assessment)

    print("\nGenerate .txt and .pdf reports")
    report_paths = create_disability_report(disability_assessment)

    if report_paths:
        print(f"Reports generated successfully: {report_paths}")

    formatted_likelihood = {}
    for scen_name, obj in all_valid_scenarios_data.items():
        p_id = obj["person"]

        if p_id in disability_assessment:
            formatted_likelihood[scen_name] = {
                'status': disability_assessment[p_id]['status'],
                'score': disability_assessment[p_id]['final_score'],
            }
    
    if formatted_likelihood:
        print("\nGenerating disability annotation plot...")
        plot_disability_annotations(all_valid_scenarios_data, formatted_likelihood)

    
    if disability_assessment:
        print("\nBehavioral assessment results:")
        for p_id, result in disability_assessment.items():
            print(f" Person: {p_id}, Status: {result['status']}, Final Score: {result['final_score']}, Cluster(K-means): {result['cluster']}")

        save_detailed_results(disability_assessment, person_aggregated_data)

    try:
        patterns = detect_disability_patterns_unsupervised(person_aggregated_data)
        print("\nUnsupervised pattern detected and saved")
    except Exception as e:
        print(f"\nError during unsupervised pattern detection: {e}")

    csv_ground_truth_path = "data/ground_truth/ground_truth.csv"
    print("\n Validation against Ground Truth")
    ground_truth = load_ground_truth(csv_ground_truth_path)

    if ground_truth:

        y_pred_bin = []
        y_true_bin = []
        valid_p_ids = []

        THRESHOLD = getattr(config, 'DISABILITY_THRESHOLD', 0.55)

        for p_id, result in disability_assessment.items():
            if p_id in ground_truth:
                valid_p_ids.append(p_id)

                true_val = ground_truth[p_id]
                try:
                    true_val = int(true_val)
                except (ValueError, TypeError):
                    print(f"Warning: Ground truth label for {p_id} is not numeric: {true_val}. Skipping.")
                    continue

                y_true_bin.append(true_val)

                score = result['final_score']
                if score >= THRESHOLD:
                    pred_val = 1
                else:
                    pred_val = 0

                y_pred_bin.append(pred_val)

                match_status = "OK" if true_val == pred_val else "WRONG"
                print(f"{p_id:15} | {true_val:<12} | {score:<15.4} | {pred_val} ({match_status})")
                
        if y_true_bin:
            matches = 0
            for i in range(len(y_true_bin)):
                true = y_true_bin[i]
                pred = y_pred_bin[i]

                if true == pred:
                    matches += 1
            accuracy = (matches / len(y_true_bin)) * 100
            print(f"\nAccuracy: {accuracy:.2f}%")

            calculate_and_plot_metrics(y_true_bin, y_pred_bin, labels=[0,1])
            export_evaluation_summary(y_true_bin, y_pred_bin)
        else:
             print("No matching Person IDs found between CSV and algorithm results.")
    else:
        print("Skipping validation: Ground truth CSV not found or empty.")
else:
    print("No valid data available for clustering and disability analysis.")
    