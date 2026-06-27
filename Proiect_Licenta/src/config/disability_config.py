"""
Global configuration parameters for the VR scenario analysis system.

This module defines all constants required for:
    - scenario validation,
    - disability score computation,
    - clustering algorithms,
    - feature extraction,
    - anomaly detection,
    - report generation and visualization.

"""

MIN_SCENARIOS_REQUIRED = 2 
MAX_CORRUPTION_RATE = 0.5

DISABILITY_WEIGHTS = {
    'distance': 0.35,
    'mahalanobis': 0.24,
    'consistency': 0.41,
}

USE_DATA_DRIVEN_THRESHOLDS = False

DISABILITY_SCORE_THRESHOLDS = {
    'HIGH':   0.8226,
    'MEDIUM': 0.6503,
    'LOW':    0.5528,
}
DATA_DRIVEN_THRESHOLD_QUANTILES = {
    'LOW': 0.25,
    'MEDIUM': 0.5,
    'HIGH': 0.75,
}

DISABILITY_THRESHOLD = 0.6503336833523391

BUFFER_START_END = 10.0
OUTPUT_DIRS = ["data/output/debug", "data/output/dendrograms", "data/output/disability_3d", "data/output/disability_reports", "data/output/evaluation_results", "data/output/plots", "data/output/plots_comparison", "data/output/text_reports"]

AGGLOMERATIVE_N_CLUSTERS = 3

KMEANS_N_CLUSTERS = 3
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42
SILHOUETTE_K_RANGE = (2, 5)

FEATURES_ENTROPY_BINS = 5
FEATURE_WINDOW_SIZE = 5

FEATURE_PAUSE_BASE_THRESHOLD = 0.01

FEATURE_SHARP_TURN_DEGREES = 30
FEATURE_AUTOCORR_MIN_FRAMES = 10

VARIATION_HIGH_THRESHOLD = 0.7
VARIATION_MEDIUM_THRESHOLD = 0.4

CLUSTER_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

DENDROGRAM_OUTPUT_DIR = "data/output/dendrograms"

RESULTS_PATH = "data/output/disability_results/behavioral_classification.json"
GROUND_TRUTH_PATH = "data/ground_truth/ground_truth.csv"