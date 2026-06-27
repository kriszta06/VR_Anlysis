from pathlib import Path
import os
import numpy as np
import csv
import json
import seaborn as sns
from sklearn.cluster import KMeans
from src.core.processing.data_loader import load_head_data
from src.core.processing.feature_extractor import extract_behavior_features
from src.core.analysis.clustering import perform_agglomerative_clustering
from src.core.analysis.disability_engine import analyze_person_disability, detect_disability_patterns_unsupervised, save_detailed_results
from src.utils.file_management import group_files_by_person
from src.config import disability_config as config
from src.visualization.plotter_3d import plot_clusters, plot_disability_annotations
from src.visualization.reports import create_visual_disability_report, plot_person_dendrogram, create_disability_report 
from src.visualization.scenario_comparison import plot_all_scenarios_comparison, plot_person_scenario_comparison
from src.evaluation.ground_truth_handler import load_ground_truth, sync_data
from src.evaluation.metrics import get_output_path, calculate_and_plot_metrics, export_evaluation_summary
from src.loocv import run_loocv, print_loocv_summary, save_loocv_results
from src.mann_whitney import run_mann_whitney

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
json_files = list((BASE_DIR / "data" / "vr_recordings").glob("*.json"))
grouped = group_files_by_person(json_files)

person_aggregated_data = {}
all_valid_scenarios_data = {}
hierarchical_data = {}

kmeans_n_clusters = getattr(config, 'KMEANS_N_CLUSTERS', 3)
kmeans_n_init = getattr(config, 'KMEANS_N_INIT', 10)
kmeans_random_state = getattr(config, 'KMEANS_RANDOM_STATE', 42)

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
            kmeans = KMeans(
                n_clusters=kmeans_n_clusters,
                n_init=kmeans_n_init,
                random_state=kmeans_random_state
            )
            labels = kmeans.fit_predict(positions)
            plot_clusters(positions, labels, scenario_name, "Head_Position")

    if hierarchical_data[person]:
        print(f"Generate comparison between scenarios for {person}")
        plot_person_scenario_comparison(hierarchical_data[person], person)

    actual_valid_scenarios = len(person_features)
    corruption_rate = (total_scenarios - actual_valid_scenarios) / total_scenarios if total_scenarios > 0 else 1
    
    print(f"Person: {person}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Valid scenarios: {actual_valid_scenarios}")
    print(f"Corruption rate: {corruption_rate:.2f}")
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

    csv_ground_truth_path = str(BASE_DIR / "data" / "ground_truth" / "ground_truth.csv")
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
                if score > THRESHOLD:
                    pred_val = 1
                else:
                    pred_val = 0

                y_pred_bin.append(pred_val)

                match_status = "OK" if true_val == pred_val else "WRONG"
                print(f"{p_id:15} | {true_val:<12} | {score:<15.4} | {pred_val} ({match_status})")
                
        if y_true_bin:
            matches = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == p)
            accuracy = (matches / len(y_true_bin)) * 100
            print(f"\nAccuracy: {accuracy:.2f}%")

            calculate_and_plot_metrics(y_true_bin, y_pred_bin, labels=[0,1])
            export_evaluation_summary(y_true_bin, y_pred_bin)
        else:
             print("No matching Person IDs found between CSV and algorithm results.")
    else:
        print("Skipping validation: Ground truth CSV not found or empty.")

    if person_aggregated_data and ground_truth:
 
        print("\n" + "="*60)
        print("Running LOOCV analysis")
        print("="*60)

        gt_for_loocv = {}
        for k, v in ground_truth.items():
            try:
                gt_for_loocv[k] = int(v)
            except (ValueError, TypeError):
                pass
    
        loocv_results = run_loocv(
            person_aggregated_data=person_aggregated_data,
            ground_truth=gt_for_loocv,
            threshold_method="youden",
        )

        if loocv_results:
            print_loocv_summary(loocv_results)
            save_loocv_results(loocv_results, output_dir=DATA_DIR/"loocv")

    print("\n" + "="*60)
    print("Running Mann-Whitney analysis")
    print("="*60)
    
    if disability_assessment and ground_truth:
        mw_scores = {
            p_id.replace("Person_", ""): result["final_score"]
            for p_id, result in disability_assessment.items()
        }
        run_mann_whitney(
            person_scores=mw_scores,
            ground_truth_path=csv_ground_truth_path,
            output_dir=DATA_DIR/"mann_whitney"
            )
else:
    print("No valid data available for clustering and disability analysis.")
    