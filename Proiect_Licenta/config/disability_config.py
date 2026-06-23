MIN_SCENARIOS_REQUIRED = 2 
MAX_CORRUPTION_RATE = 0.5

DISABILITY_WEIGHTS = {
    'distance': 0.35,
    'mahalanobis': 0.24,
    'consistency': 0.41,
}

USE_DATA_DRIVEN_THRESHOLDS = False

DISABILITY_DATA_DRIVEN_THRESHOLDS = {
    'HIGH': 0.72,
    'MEDIUM': 0.61,
    'LOW': 0.52,
}
DATA_DRIVEN_THRESHOLD_QUANTILES = {
    'LOW': 0.25,
    'MEDIUM': 0.5,
    'HIGH': 0.75,
}

DISABILITY_THRESHOLD = 0.66

BUFFER_START_END = 10.0
OUTPUT_DIRS = ["data/output/debug", "data/output/dendrograms", "data/output/disability_3d", "data/output/disability_reports", "data/output/evaluation_results", "data/output/plots", "data/output/plots_comparison", "data/output/text_reports"]
