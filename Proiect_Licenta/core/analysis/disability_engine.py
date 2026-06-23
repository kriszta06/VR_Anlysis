import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
from core.processing.feature_extractor import extract_behavior_features
from core.processing.data_loader import load_head_data
from utils.file_management import group_files_by_person
from config import disability_config as config
from matplotlib import pyplot as plt
import json
import os

def find_optimal_k_silhouette(data, k_range=(2, 5)):
    """
    Find optimal number of clusters using Silhouette Score.
    
    Parameters:
    :param data: Feature matrix (n_samples, n_features)
    :param k_range: Tuple (min_k, max_k) for search range
    
    Returns:
    dict with 'optimal_k', 'silhouette_scores', 'score_values'
    """
    n_samples = data.shape[0]
    if n_samples < 2:
        return {'optimal_k': 1, 'silhouette_scores': {}, 'note': 'Single sample, cluster = 1'}
    if n_samples == 2:
        return {'optimal_k': 2, 'silhouette_scores': {}, 'note': 'Two samples, cluster = 2'}

    k_max = min(k_range[1], n_samples)
    k_min = max(k_range[0], 2)
    
    silhouette_scores = {}
    
    for k in range(k_min, k_max + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores[k] = score
        except Exception as e:
            print(f"Warning: Could not compute silhouette score for k={k}: {e}")
            continue
    
    if not silhouette_scores:
        # Fallback to k=3 if silhouette calculation fails
        return {'optimal_k': min(3, n_samples), 'silhouette_scores': {}, 'note': 'Fallback to k=3'}
    
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    
    return {
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'best_score': silhouette_scores[optimal_k]
    }


def get_disability_status(disability_score, thresholds=None):
    """
    Converts a normalized disability score into a categorical status.

    Parameters:
    :param disability_score: float, final normalized score
    :param thresholds: dict or None, score boundaries for categories
    :return: 'HIGH', 'MEDIUM', 'LOW', or 'NONE'
    """
    if thresholds is None:
        thresholds = getattr(config, 'DISABILITY_SCORE_THRESHOLDS', {
            'HIGH': 0.72, 'MEDIUM': 0.61, 'LOW': 0.52
        })

    high_thr = thresholds.get('HIGH', 0.72)
    medium_thr = thresholds.get('MEDIUM', 0.61)
    low_thr = thresholds.get('LOW', 0.52)

    if not (high_thr >= medium_thr >= low_thr):
        print("Warning: Threshold values are not ordered correctly. Reverting to default thresholds.")
        cfg = getattr(config, 'DISABILITY_SCORE_THRESHOLDS', {})
        high_thr = cfg.get('HIGH', 0.72)
        medium_thr = cfg.get('MEDIUM', 0.61)
        low_thr = cfg.get('LOW', 0.52)
        high_thr, medium_thr, low_thr = 0.7, 0.5, 0.3

    if disability_score >= high_thr:
        return 'HIGH'
    if disability_score >= medium_thr:
        return 'MEDIUM'
    if disability_score >= low_thr:
        return 'LOW'
    return 'NONE'


def compute_data_driven_thresholds(final_scores, quantiles=None):
    """
    Computes LOW/MEDIUM/HIGH thresholds from observed final scores.

    Parameters:
    :param final_scores: array-like of numeric scores
    :param quantiles: dict with 'LOW', 'MEDIUM', 'HIGH' quantiles
    :return: dict with numeric threshold values
    """
    if quantiles is None:
        quantiles = getattr(config, 'DATA_DRIVEN_THRESHOLD_QUANTILES', {
            'LOW': 0.25,
            'MEDIUM': 0.5,
            'HIGH': 0.75,
        })

    valid_scores = np.asarray(final_scores, dtype=float)
    valid_scores = valid_scores[np.isfinite(valid_scores)]
    if valid_scores.size == 0:
        return {
            'HIGH': 0.7,
            'MEDIUM': 0.5,
            'LOW': 0.3,
        }

    low_q = min(max(quantiles.get('LOW', 0.25), 0.0), 1.0)
    med_q = min(max(quantiles.get('MEDIUM', 0.5), 0.0), 1.0)
    high_q = min(max(quantiles.get('HIGH', 0.75), 0.0), 1.0)

    if not (low_q <= med_q <= high_q):
        low_q, med_q, high_q = 0.25, 0.5, 0.75

    thresholds = {
        'LOW': float(np.quantile(valid_scores, low_q)),
        'MEDIUM': float(np.quantile(valid_scores, med_q)),
        'HIGH': float(np.quantile(valid_scores, high_q)),
    }

    thresholds['LOW'] = min(thresholds['LOW'], thresholds['MEDIUM'], thresholds['HIGH'])
    thresholds['MEDIUM'] = min(max(thresholds['MEDIUM'], thresholds['LOW']), thresholds['HIGH'])
    thresholds['HIGH'] = max(thresholds['HIGH'], thresholds['MEDIUM'], thresholds['LOW'])

    return thresholds


def analyze_person_disability(all_person_features):
    '''
    Analzye behavioral deviation of each person based on extracted feature vectors
    and compute a composite disability score. 
    This function performs:
    1. Standardization of feature vectors
    2. Dimensionality reduction using PCA (95% explained variance)
    3. K-means clustering for behavioral pattern identification
    4. Distance-based anomaly scoring
    5. Mahalanobis distance computation
    6. Behavioral consistency measurment
    7. Final weighted disability scoring

    Parameters:
    
    :param all_person_features: dict
    Dictionary mapping:
    person_id (str) -> feature_vector (list or numpy array)
    Each feature vector must have the same length and represent
    behavioral characterstics extracted for the participant.

    Returns:

    dict
    Dictionary mapping each person_id to a dictionary containing:
    -cluster(int):
    Cluster label assigned by K-means clustering
    -distance_score(float):
    Euclidean distance from the global centroid in PCA space. 
    Measures global deviation from average behavior.
    -mahalanobis_score(float):
    Multivariate distance considering feature covariance.
    Measures statistical abnormality relative to group distribution.
    -consistency_score(float):
    Average distance to all other participants.
    Measures how behaviorally similar/different the person is
    compared to teh rest of the group
    -final_score(float):
    Weight composite score combining distance, Mahalanobis, and consistency.
    Weights are configurable via the disability config and the values are
    normalized between 0 and 1.
    -status(str):
    Categorical disability assessment: "NONE" (minimal deviation), "LOW" (slight deviation), "MEDIUM" (moderate deviation), "HIGH" (strong behavioral deviation)

    Note:

    -PCA reduces noise and removes redundant correlated features.
    -K-means identifies behavioral clusters based on feature similarity.
    -Mahalanobis distance is computed in PCA-reduced space using a robust pseudoinverse.
    -The final score represents overall behavioral abnormality relative
    to the group distribution.
    
    '''

    feature_vectors = []
    person_ids = []

    for person_id, features in all_person_features.items():
        feature_vectors.append(features)
        person_ids.append(person_id)

    feature_matrix = np.array(feature_vectors)
    n_persons = len(person_ids)

    if n_persons == 0:
        return {}

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    if n_persons == 1:
        return {
            person_ids[0]: {
                'cluster': 0,
                'distance_score': 0.0,
                'mahalanobis_score': 0.0,
                'consistency_score': 0.0,
                'final_score': 0.0,
                'status': 'NONE'
            }
        }

    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(scaled_features)

    # Use Silhouette Score to find optimal K
    silhouette_result = find_optimal_k_silhouette(principal_components, k_range=(2, 5))
    n_clusters = silhouette_result['optimal_k']
    
    print(f"\n[Silhouette Analysis] Optimal clusters: {n_clusters}")
    if 'silhouette_scores' in silhouette_result and silhouette_result['silhouette_scores']:
        print(f"  Silhouette scores: {silhouette_result['silhouette_scores']}")
        print(f"  Best score: {silhouette_result['best_score']:.4f}")
    
    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clustering.fit_predict(principal_components)

    centroid = np.mean(principal_components, axis=0)
    distances = cdist(principal_components, [centroid]).flatten()

    try:
        cov_matrix = np.cov(principal_components.T)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        mahalanobis_scores = []
        pc_mean = np.mean(principal_components, axis=0)
        for i in range(n_persons):
            diff = principal_components[i] - pc_mean
            score = np.sqrt(diff.T @ inv_cov_matrix @ diff)
            mahalanobis_scores.append(score if np.isfinite(score) else distances[i])

        mahalanobis_scores = np.array(mahalanobis_scores)
    except Exception:
        mahalanobis_scores = distances

    consistency_scores = []
    for i in range(n_persons):
        other_features = np.vstack([scaled_features[j] for j in range(len(scaled_features)) if j != i])
        avg_dist = np.mean(cdist([scaled_features[i]], other_features))
        consistency_scores.append(avg_dist)
        
    consistency_scores = np.array(consistency_scores)

    def robust_normalie(arr):
        "Z-score then sigmoid [0,1]. Robust to single extreme outliers"
        mu = np.mean(arr)
        sig = np.std(arr) if np.std(arr) > 0 else 1.0
        z = (arr -mu) / sig

        return 1.0 / (1.0 + np.exp(-z))
    
    norm_distances = robust_normalie(distances)
    norm_mahals = robust_normalie(mahalanobis_scores)

    cluster_sizes = np.bincount(labels)
    majority_label = np.argmax(cluster_sizes)
    majority_mask = labels == majority_label
    healthy_centroid = np.mean(scaled_features[majority_mask], axis=0)
    dist_to_healthy = np.linalg.norm(scaled_features - healthy_centroid, axis=1)
    norm_consistency = robust_normalie(dist_to_healthy)

    final_scores = []
    normalized_scores = []

    weights = getattr(config, 'DISABILITY_WEIGHTS', {'distance': 0.5, 'mahalanobis': 0.3, 'consistency': 0.2})
    weight_distance = weights.get('distance', 0.5)
    weight_mahal = weights.get('mahalanobis', 0.3)
    weight_consistency = weights.get('consistency', 0.2)

    for i in range(n_persons):
        nd = norm_distances[i]
        nm = norm_mahals[i]
        nc = norm_consistency[i]

        nd = 0.0 if not np.isfinite(nd) else nd
        nm = 0.0 if not np.isfinite(nm) else nm
        nc = 0.0 if not np.isfinite(nc) else nc

        score = (
            weight_distance * nd
            + weight_mahal * nm
            + weight_consistency * nc
        )
        score = 0.0 if not np.isfinite(score) else score

        final_scores.append(score)

        normalized_scores.append({
            'norm_distance': nd,
            'norm_mahal': nm,
            'norm_consistency': nc,
        })

    final_scores = np.array(final_scores)

    disability_assessment = {}
        
    thresholds = getattr(config, 'DISABILITY_SCORE_THRESHOLDS', {
        'HIGH': 0.72,
        'MEDIUM': 0.61, 
        'LOW': 0.52
    })

    for i, person_id in enumerate(person_ids):
        disability_score = final_scores[i]
        status = get_disability_status(disability_score, thresholds)

        final_score_clean = disability_score
        if np.isnan(final_score_clean) or np.isinf(final_score_clean):
            final_score_clean = 0.0
            
        distance_score_clean = distances[i]
        if np.isnan(distance_score_clean) or np.isinf(distance_score_clean):
            distance_score_clean = 0.0

        mahal_score_clean = mahalanobis_scores[i]
        if np.isnan(mahal_score_clean) or np.isinf(mahal_score_clean):
            mahal_score_clean = 0.0

        consistency_score_clean = consistency_scores[i]
        if np.isnan(consistency_score_clean) or np.isinf(consistency_score_clean):
            consistency_score_clean = 0.0
            
        disability_assessment[person_id] = {
            'cluster': int(labels[i]),
            'distance_score': float(distances[i]),
            'mahalanobis_score': float(mahalanobis_scores[i]),
            'consistency_score': float(consistency_scores[i]),
            'final_score': float(disability_score),
            'status': status
        }

    return disability_assessment

def detect_disability_patterns_unsupervised(all_scenarios):
    '''
    Perform unsupervised behavioral pattern detection across multiple scenarios.

    This function:
    1. Extracts behavioral feature vectors for each scenario
    2. Standardizes the features
    3. Applies PCA for dimensionality reduction (95% explained variance)
    4. Performs K-means clustering for behavioral pattern identification
    5. Computes deviation from the global behavioral centroid
    6. Generates and saves a PCA visualization of behavioral pattern space

    Parameters:
    
    :param all_scenarios: dict
    Dictionary mapping: scenario_name (str) -> raw behavioral data
    Each scenario is processed through 'extract_behavioral_features' 
    to generate a numerical feature vector of equal length.

    Returns:

    dict
    Dictionary mapping each scenario_name to a dictinary containing:
    -cluster(int):
    Cluster label assigned by K-means clustering
    -distance(float):
    Euclidean distance from the global centroid in PCA-reduced 
    feature space. Represents overall behavioral deviation.
    -score(float):
    Normalized deviation score (0-1), computed relative to the 
    maximum observed distance.
    -variation_level(str):
    Behavioral variation severity:
    "LOW" = small deviation
    "MEDIUM" = moderate deviation
    "HIGH" = strong deviation

    Note:
    -PCA reduces dimensionality while preserving 95% of the variance, 
    improving clustering stability.
    -K-means identifies behavioral groupings based on feature similarity
    -The centroid distance reflects how behaviorally different 
    a scenario is compared to the overall average behavior.
    -A visualization is saved as "behavioral_pattern_space.png'

    '''

    feature_vectors = []
    scenario_names = []

    for name, data in all_scenarios.items():
        if isinstance(data, np.ndarray) and data.ndim == 1:
            features = data
        else:
            features = extract_behavior_features(data)
        feature_vectors.append(features)
        scenario_names.append(name)

    feature_matrix = np.array(feature_vectors)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(scaled_features)

    # Use Silhouette Score to find optimal K
    silhouette_result = find_optimal_k_silhouette(principal_components, k_range=(2, 5))
    n_clusters = silhouette_result['optimal_k']
    
    print(f"[Silhouette Analysis] Optimal clusters for unsupervised: {n_clusters}")
    if 'silhouette_scores' in silhouette_result and silhouette_result['silhouette_scores']:
        print(f"  Silhouette scores: {silhouette_result['silhouette_scores']}")
    
    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clustering.fit_predict(principal_components)

    centroid = np.mean(principal_components, axis=0)
    distances = cdist(principal_components, [centroid]).flatten()

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)

    # Handle case where PCA results in 1D
    if principal_components.shape[1] == 1:
        # 1D PCA - plot as horizontal scatter
        for label in unique_labels:
            mask = labels == label
            plt.scatter(principal_components[mask, 0], np.zeros(np.sum(mask)), 
                       s=150, alpha=0.7, label=f'Group {label}')
                
        for i, name in enumerate(scenario_names):
            plt.annotate(f"{name}\n({distances[i]:.2f})",
                         (principal_components[i, 0], 0),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center',
                         fontsize=9)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Dummy Y', fontsize=12)
    else:
        # 2D+ PCA - normal 2D plot
        for label in unique_labels:
            mask = labels == label
            plt.scatter(principal_components[mask, 0], principal_components[mask, 1],
                        s=150, alpha=0.7, label=f'Group {label}')
                
        for i, name in enumerate(scenario_names):
            plt.annotate(f"{name}\n({distances[i]:.2f})",
                         (principal_components[i, 0], principal_components[i, 1]),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center',
                         fontsize=9)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
    
    # plt.title('Behavioral Pattern Space (PCA)', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.2)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              'data', 'output', 'disability_results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'behavioral_pattern_space.png'), dpi=300, bbox_inches='tight')
    plt.close()

    behavioral_variation = {}
    max_distance = np.max(distances)

    for i, name in enumerate(scenario_names):
        variation_score = float(distances[i] / max_distance)
        variation_level = 'HIGH' if variation_score > 0.7 else 'MEDIUM' if variation_score > 0.4 else 'LOW'

        behavioral_variation[name] = {
            'cluster': int(labels[i]),
            'distance': float(distances[i]),
            'score': variation_score,
            'variation_level': variation_level
        }
    
    return behavioral_variation

def save_detailed_results(disability_assessment, all_person_features):
    '''
    Save detailed behavioral disability analysis results to a structured JSON file.

    This function:
    1. Creates (if necessary) an output directory inside:
    data/output/disability_results
    2. Builds a structured JSON report containing:
    -Analysis metadata
    -Methodological description
    -Per-person behavioral classification results
    -Raw feature vectors
    3. Exports the results as 'behavioral_classification.json'

    Paramteres:
    
    :param disability_assessment: dict
    Dictionary returned by 'analyze_person_disability',
    mapping:
    person_id -> {
        'cluster':int.
        'distance_score':float,
        'mahalanobis_score':float,
        'consistency_score':float,
        'final_score':float,
        'status':str
    }
    Contains computed behavioral deviation metrics and 
    severity classification for each person.

    :param all_person_features: dict
    Dictionary mapping:
    person_id (str) -> numpy array (feature vector)
    Contains the original extracted behavioral features
    for each participant.

    Output:
    A JSON file saved to:
    data/output/disability_results/behavioral_classification.json

    The file includes:
    -analysis_timestamp
    -total_persons
    -methodology description
    -per-person detailed results
    -raw feature vectors

    Note:
    -The JSON file provides full reproducibility by storing
    both computed scores and original feature values.
    -Threshold group assignments are derived from either quantile-based
    data-driven values or default score boundaries.
    -The output directory is created automatically if missing.
    '''

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              'data', 'output', 'disability_results')
    os.makedirs(output_dir, exist_ok=True)

    weights = getattr(config, 'DISABILITY_WEIGHTS', {'distance': 0.5, 'mahalanobis': 0.3, 'consistency': 0.2})
    thresholds = getattr(config, 'DISABILITY_SCORE_THRESHOLDS', {'HIGH': 0.72, 'MEDIUM': 0.61, 'LOW': 0.52})
    score_description = (
        f"Combined: {weights.get('distance', 0.5) * 100:.0f}% distance + "
        f"{weights.get('mahalanobis', 0.3) * 100:.0f}% Mahalanobis + "
        f"{weights.get('consistency', 0.2) * 100:.0f}% consistency"
    )

    detailed_results = {
        'analysis_timestamp': str(np.datetime64('now')),
        'total_persons': len(disability_assessment),
        'methodology': {
            'pca_components': '95% variance retained',
            'clustering': 'K-means with Silhouette-optimized K (2-5 range)',
            'scoring': score_description,
            'thresholds': {
                'HIGH': f">{thresholds.get('HIGH', 0.7)}",
                'MEDIUM': f">{thresholds.get('MEDIUM', 0.5)}",
                'LOW': f">{thresholds.get('LOW', 0.3)}",
                'NONE': f"<={thresholds.get('LOW', 0.3)}"
            }
        },
        'persons': {}
    }

    for person_id, result in disability_assessment.items():
        detailed_results['persons'][person_id] = {
            'behavioral_group': result['status'],
            'final_score': result['final_score'],
            'cluster': result['cluster'],
            'distance_score': result['distance_score'],
            'mahalanobis_score': result['mahalanobis_score'],
            'consistency_score': result['consistency_score'],
            'raw_features': all_person_features[person_id].tolist()
        }
    
    file_path = os.path.join(output_dir, 'behavioral_classification.json')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print("\n Results saved as: behavioral_classification.json")


# if __name__ == "__main__":
#     print("DEBUG: Running disability_engine workflow for first 3 persons")

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
#     hierarchical_data = {}

#     for person in selected_persons:
#         files = grouped.get(person, [])
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
#             scenario_type = file_path.stem
#             hierarchical_data[person][scenario_type] = scenario_data

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
#             print("Aggregated features for person:")
#             print(person_aggregated_data[person])
#         else:
#             print(f"{person} is excluded from analysis due to insufficient valid scenarios or high corruption rate")

#     if not person_aggregated_data:
#         print("\nNo valid person data available for disability analysis.")
#     else:
#         print(f"\nRunning disability analysis for {len(person_aggregated_data)} persons")
#         disability_assessment = analyze_person_disability(person_aggregated_data)
#         save_detailed_results(disability_assessment, person_aggregated_data)

#         print("\nBehavioral disability assessment results:")
#         for p_id, result in disability_assessment.items():
#             print(f" Person: {p_id}, Status: {result['status']}, Final Score: {result['final_score']:.4f}, Cluster: {result['cluster']}")
    

#     print("\nTesting detect_disability_patterns_unsupervised per participant")

#     for person, scenarios in hierarchical_data.items():
#         if not scenarios:
#             print(f"No scenarios found for {person}, skipping unsupervised pattern detection.")
#             continue

#         print(f"\nRunning detect_disability_patterns_unsupervised for {person}")
#         try:
#             pattern_results = detect_disability_patterns_unsupervised(scenarios)
#             print(f"Disability pattern detection results for {person}:")
#             for scenario_name, result in pattern_results.items():
#                 print(f"  {scenario_name}: cluster={result['cluster']}, score={result['score']:.3f}, variation_level={result['variation_level']}")
#         except Exception as e:
#             print(f"Error during detect_disability_patterns_unsupervised for {person}: {e}")