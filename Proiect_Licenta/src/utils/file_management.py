from pathlib import Path
import re
import shutil
import numpy as np
import os
import glob

def group_files_by_person(json_files):
    '''
    Groups JSON files by person based on filename pattern.

    Expected filename pattern:
        <number>-<letters>.json

    where <number> identifies the person and <letters> is one or more
    alphabetic characters.
    
    The letter has to be one of the next: 'A', 'B' or 'C'

    
    Parameters: 
    ----------

    json_files: list of pathlib.Path 
        List of file paths to JSON files.

    Returns:
    --------

    dict[str, list[pathlib.Path]]
        Dictionary mapping person IDs to lists of file paths.
    Example:
    {
        "Person_1": [Path("1-A.json"), Path("1-B.json")], 
        "Person_2": [Path("2-A.json")], 
        …
    }

    Note:
    -------

    - If the filename matches the expected pattern, it will be grouped by person ID.
    - Filenames that do not match the expected pattern are skipped and a
      debug message is printed.
    - Person IDs are strings of the form ``Person_<number>``.
    '''

    grouped = {}
    for file_path in json_files:

        match = re.search(r'(\d+)-[ABC]+\.json$', file_path.name)
        if match:
            person_id = f"Person_{match.group(1)}"
            if person_id not in grouped:
                grouped[person_id] = []
            grouped[person_id].append(file_path)
        else:
            print(f"DEBUG: Filename {file_path.name} does not match expected pattern.")

    print(f"DEBUG: Grouped files by person: {grouped}")
    for person, files in grouped.items():
        print(f"DEBUG: {person} has {len(files)} files: {[file.name for file in files]}")
    
    return grouped

def combine_features_for_person(scenario_features, expected_scenarios, features_per_scenario):
    """
    Combines feature vectors extracted from multiple scenarios of a single
    participant into a unified feature vector.

    The resulting feature vector preserves the order specified by
    ``expected_scenarios``. If features for an expected scenario are missing,
    a zero vector of the appropriate length is inserted. Any NaN or infinite
    values in the extracted features are replaced with zeros to ensure
    numerical stability.

    Parameters
    ----------
    scenario_features : dict[str, array-like]
        Dictionary mapping scenario names to their corresponding feature
        vectors.

        Example:
            {'1-A': [f1, f2, f3], '1-B': [f1, f2, ...]}

    expected_scenarios : list[str]
        List of expected scenario names. The order of this list determines
        the order in which scenario feature vectors are concatenated.

    features_per_scenario : dict[str, int]
        Dictionary mapping each scenario name to the expected number of
        features. This information is used to generate zero vectors for
        missing scenarios.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the concatenated feature vectors for
        all expected scenarios. Missing scenarios are represented by zero
        vectors, and any NaN or infinite values are replaced with zeros.

    Notes
    -----
    - If ``scenario_features`` is empty, an empty NumPy array is returned.
    - All feature vectors are flattened before concatenation.
    - Missing scenarios are padded with zeros to preserve a consistent
      feature dimensionality across all participants.
    """

    if not scenario_features:
        return np.array([])

    combined = []

    for scen in expected_scenarios:
        if scen in scenario_features and scenario_features[scen] is not None:
            arr = np.asarray(scenario_features[scen]).flatten()
        else:
            arr = np.zeros(features_per_scenario[scen], dtype=float)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        combined.extend(arr.tolist())

    return np.array(combined, dtype=float)

def cleanup_disability_files():
    """
    Removes previously generated analysis results, reports, visualizations, and
    temporary files from the output directory.

    The function deletes files and directories produced during earlier executions
    of the analysis pipeline to ensure that subsequent runs generate a clean and
    consistent set of results. Only generated output files matching predefined
    patterns are removed; original input data remain unchanged.

    The cleanup includes, but is not limited to:

    - Participant similarity dendrograms.
    - Individual 3D trajectory visualizations.
    - Behavioral variation reports.
    - Scenario comparison visualizations.
    - Cluster debugging figures and statistics.
    - Timestamped text and PDF reports.
    - Evaluation results (e.g., confusion matrices and summary files).
    - Temporary debug files.
    - Generated plots and disability analysis results.

    Notes
    -----
    - If the output directory does not exist, the function terminates without
    performing any action.
    - Files and directories are removed recursively according to predefined
    filename patterns.
    - Any exceptions raised during file deletion are caught and reported through
    debug messages, allowing the cleanup process to continue.
    - Original input datasets are never modified or deleted.
    """

    base_dr = Path(__file__).resolve().parent.parent
    output_dir = base_dr / "data" / "output"

    print(f"DEBUG: Base directory for cleanup: {base_dr}")
    print(f"DEBUG: Output directory for cleanup: {output_dir}")

    if not output_dir.exists():
        print(f"DEBUG: Output directory does not exist:", output_dir)
        return 

    file_patterns = {
        '**/*participant_similarity_dendrogram.png',
        '**/*Person_*_3d_path.png',
        '**/*behavioral_variation_report.png',
        '**/*Person_*_scenarios_comparison.png',
        '**/*cluster_DEBUG_SCENARIO_head-eyes.png',
        '**/*clusters_DEBUG_SCENARIO_head-eyes_stats.json',
        '**/*comparison_*_scenarios_3d.png',
        '**/*behavioral_report_*.txt',
        '**/*behavioral_report_*.pdf',
        '**/debug/debug_*.txt',
        '**/evaluation_results/confusion_matrix.png',
        '**/evaluation_results/evaluation_summary.json',
        '**/dendrogram/*.png',
        '**/disability_results/*',
        '**/plots/*'
    }

    print("Clean disability files and scenario comparison")

    for file_pattern in file_patterns:
        matching_files = glob.glob(str(output_dir / file_pattern), recursive=True)
        print(f"{file_pattern} -> {matching_files}")
        for file_path in matching_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted folder: {file_path}")
            except Exception as e:
                print(f"Error occured at deletion of {file_path}: {e}")
    
    print("Initial files are kept")
    print("Comparison files will be regenerated")


# if __name__ == "__main__":
#     print("DEBUG : Running file management script + feature combination script")
#     test_files = [
#         Path("data/vr_recordings/1-A.json"),
#         Path("data/vr_recordings/1-B.json"),
#         Path("data/vr_recordings/1-C.json"),
#         Path("data/vr_recordings/2-A.json"),
#         Path("data/vr_recordings/2-B.json"),
#         Path("data/vr_recordings/2-C.json"),
#         Path("data/vr_recordings/3-A.json"),
#         Path("data/vr_recordings/3-B.json"),
#         Path("data/vr_recordings/3-C.json")
#     ]

#     print(f"Total test files: {len(test_files)}")
#     for f in test_files:
#         print(f" - {f.name}")
    
#     grouped = group_files_by_person(test_files)

#     print("\nGrouped files:")
#     for person, files in grouped.items():
#         print(f"{person}: {[file.name for file in files]}")

#     unmatched_files = [f for f in test_files if not re.search(r'(\d+)-[A-Za-z]+\.json$', f.name)]
#     if unmatched_files:
#         print("\nFiles that did not match the expected pattern:")
#         for f in unmatched_files:
#             print(f" - {f.name}")

# print("\nDebug: Testing combine_features_for_person")

# expected_scenarios = ["scenario_A", "scenario_B", "scenario_C"]
# features_for_person = {
#     "scenario_A": 5,
#     "scenario_B": 3,
#     "scenario_C": 4
# }

# all_person_scenarios = {
#     "person_1": {
#         "scenario_A": np.random.rand(5),
#         "scenario_B": np.random.rand(3),
#         "scenario_C": np.random.rand(4)    
#         },
#     "person_2": {
#         "scenario_A": np.random.rand(5),
#         "scenario_C": np.random.rand(4)
#     },
#     "person_3": {
#         "scenario_B": np.random.rand(3),
#         "scenario_C": np.random.rand(4)
#     }
# }

# combined_vector = {}

# for pid, features in all_person_scenarios.items():
#     vec = combine_features_for_person(features, expected_scenarios, features_for_person)
#     combined_vector[pid] = vec
#     print(f"Person ID: {pid}")
#     print(f" Combined vector shape: {vec.shape}")
#     print(f" Combined vector: {vec}")
#     print(f" Total sum (sanity check): {np.sum(vec):.3f}\n")

# vector_lengths = [len(v) for v in combined_vector.values()]
# print("Sanity check: all vectors same length:", vector_lengths)
# if len(set(vector_lengths)) == 1:
#     print("All vectors are comparable across persons")
# else:
#     print("Vector lengths differ")

# print("\nDebug: Running cleanup of generated files")

# cleanup_disability_files()
# print("Debug: Finished cleanup of generated files\n")




