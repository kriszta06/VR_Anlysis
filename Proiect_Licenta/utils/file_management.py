from pathlib import Path
import re
import shutil
import numpy as np
import os
import glob

def group_files_by_person(json_files):
    '''
    Groups JSON files by person based on filename pattern.

    Files should follow this pattern:
    '<number>-<letters>.json' -> The letter has to be one of the next: 'A', 'B' or 'C'
    <number> identifies the person.
    
    Parameters: 

    :param json_files: list of pathlib.Path => list of file paths to JSON files. Each file path should have a '.name' attribute.

    Returns:
    
    dict: Dictionary mapping person IDs to lists of file paths.
    Example:
    {
        "Person_1": [Path("1-A.json"), Path("1-B.json")], 
        "Person_2": [Path("2-A.json")], 
        …
    }

    Note:
    - if a filename does not match the expected pattern, it will be skipped and a debug message will be printed.
    - person IDs are strings of the form 'Person_<number>'
    '''

    grouped = {}
    for file_path in json_files:

        match = re.search(r'(\d+)-[A-Za-z]+\.json$', file_path.name)
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
    '''
    Combines features from multiple scenarios of a single person into a single feature vector.
    This function ensures that each person has a complete feature vector even if
    some scenario are missing. Missing scenarios are filled with zeros.
    Special values such as: Nan or infinite values are replaced with 0.

    Parameters:
    
    :param scenario_features: dict
    Dictionary mapping scenario names to feature vectors (list or numpy array)
    extracted from that scenario.
    Example: {'1-A': [f1, f2, f3, ...], '1-B': [f1, f2, ...]}

    :param expected_scenarios: list of str
    List of scenario names that are expected for this person.
    This defines the order in which features are combined.
    Missing scenarios will be replaced with zeros.

    :param features_per_scenario: dict
    Dictionary mapping scenario names to the number of features expected
    for each scenario. Used to pad missing scenarios with zeros.

    Returns:

    numpy.ndarray
    1D array containing combined features for all scenarios. 
    The order of features corresponds to the order in 'expected_scenarios'.
    Missing scenario features are filled with zeros.
    '''

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
    '''
    Deletes previously generated report and scenario comparison files.
    This function cleans up files generated in previous runs to avoid conflicts 
    or duplication. It targets the following types of files:

    1. Participant similarity visualization:
    - 'participant_similarity_dendrogram.png'

    2. Individual 3D paths per person:
    - 'Person_*_3d_path.png' (* = person number)

    3. Behavioral variation reports:
    - 'behavioral_variation_report.png'

    4. Scenario comparison per person:
    - 'Person_*_scenarios_compariso.png' (* = person number)

    5. Cluster debug files:
    - 'cluster_DEBUG_SCENARIO_head-eyes.png'
    - 'cluster_DEBUG_SCENARIO_head-eyes_stats.json'

    6. Scenario comparison visualizations:
    - 'comparison_*_scenarios_3d.png' (* = scenario letter, could be one of the next letters: A, B, C)

    7. Timestamped behavioral reports:
    - 'behavioral_report_*.txt'
    - 'behavioral_report_*.pdf'

    Note:
    Original input files are kept.
    Any files matching the patterns above are deleted.
    Errors during deletion are caught and printed for debugging.
    This cleanup allows fresh regeneration of all analysis and report files.
    '''

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




