import csv
import json 
import os

def load_ground_truth(csv_path):
    """
    Loads clinical diagnosis labels from a CSV file and stores them in a
    dictionary indexed by participant identifier.

    The function reads the specified CSV file, validates the presence of the
    required columns (``person_id`` and ``diagnosis``), normalizes the input
    values, and converts diagnosis labels to integer values whenever possible.
    The resulting dictionary maps each participant identifier to its
    corresponding clinical diagnosis.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the ground-truth clinical diagnoses.

    Returns
    -------
    dict
        A dictionary mapping participant identifiers (e.g., ``Person_1``)
        to their corresponding diagnosis labels. If the file cannot be read
        or the required columns are missing, an empty dictionary is returned.
    """

    ground_truth = {}

    print(f"Looking for ground truth at: {os.path.abspath(csv_path)}")

    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            if reader.fieldnames:
                reader.fieldnames = [f.strip() for f in reader.fieldnames]

            header_map = {k.strip().lower(): k for k in reader.fieldnames or []}
            person_key = header_map.get('person_id')
            diagnosis_key = header_map.get('diagnosis')

            if person_key is None or diagnosis_key is None:
                raise KeyError('CSV must contain person_id and diagnosis columns')

            for row in reader:
                normalized_row = {
                    (k.strip().lower() if k else k): (v.strip() if v else '')
                    for k, v in row.items()
                }
                person_id = f"Person_{normalized_row.get('person_id', '')}"
                raw_diagnosis = normalized_row.get('diagnosis', '')

                try:
                    diagnosis = int(raw_diagnosis)
                except ValueError:
                    try:
                        diagnosis = int(float(raw_diagnosis))
                    except ValueError:
                        diagnosis = raw_diagnosis.upper()

                ground_truth[person_id] = diagnosis

        print(f"Success: Loaded {len(ground_truth)} ground truth records.")
    except KeyError as e:
        print(f"Error: Missing column in CSV: {e}.")
    except Exception as e:
        print(f"Error: Failed to read CSV file: {e}")

    return ground_truth

def sync_data(program_results, ground_truth):
    """
    Aligns predicted behavioral classifications with the corresponding
    ground-truth clinical diagnoses.

    The function matches participants present in both the algorithm output
    and the ground-truth dataset. It generates aligned lists of true labels,
    predicted labels, and participant identifiers, which can subsequently be
    used for performance evaluation and metric computation.

    Parameters
    ----------
    program_results : dict
        Dictionary containing the behavioral classification results produced
        by the algorithm.

    ground_truth : dict
        Dictionary mapping participant identifiers to their corresponding
        clinical diagnosis labels.

    Returns
    -------
    tuple[list, list, list]
        A tuple containing:
        - the ground-truth diagnosis labels (`y_true`),
        - the predicted behavioral group labels (`y_pred`),
        - the corresponding participant identifiers (`person_ids`).

        Only participants present in both input datasets are included.
    """

    y_true = []
    y_pred = []
    person_ids = []

    prog_persons = program_results.get('persons', {})

    for p_id, true_diag in ground_truth.items():
        if p_id in prog_persons:
            y_true.append(true_diag)
            y_pred.append(prog_persons[p_id].get('behavioral_group', 'UNKNOWN'))
            person_ids.append(p_id)

    print(f"Synchronized {len(y_true)} subjects for evaluation.")
    return y_true, y_pred, person_ids

# if __name__ == "__main__":
#     print("Debug: Testing ground_truth_handler.py")
#     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     sample_csv = os.path.join(repo_root, "data", "ground_truth", "ground_truth.csv")
#     gt_data = load_ground_truth(sample_csv)

#     if gt_data:
#         print(f"Successfully mapped {len(gt_data)} persons.")
#         target_persons = ["Person_1", "Person_10", "Person_11"]
#         first_persons = [pid for pid in target_persons if pid in gt_data]
#         missing_persons = [pid for pid in target_persons if pid not in gt_data]

#         if missing_persons:
#             print(f"Warning: Missing ground truth records for: {missing_persons}")

#         print("Selected persons from ground truth:")
#         for pid in first_persons:
#             print(f" {pid}: {gt_data[pid]}")
#     else:
#         print(" Error: No data loaded.")
#         first_persons = []
    
#     print("Debug: Testing sync_data function")

#     mock_program_results = {"persons": {}}
#     mock_ground_truth = {}

#     for idx, pid in enumerate(first_persons):
#         if idx == 0:
#             status = "NONE"
#             score = 0.15
#         elif idx == 1:
#             status = "HIGH"
#             score = 0.85
#         else:
#             status = "MEDIUM"
#             score = 0.55

#         mock_program_results["persons"][pid] = {
#             "behavioral_group": status,
#             "final_score": score
#         }
#         mock_ground_truth[pid] = gt_data[pid]

#     y_true, y_pred, p_ids = sync_data(mock_program_results, mock_ground_truth)

#     print("\nAligned Data Results:")
#     for i in range(len(p_ids)):
#         print(f" ID: {p_ids[i]} | True: {y_true[i]:6} | Pred: {y_pred[i]:6}")

#     if len(y_true) == len(first_persons) == 3:
#         print("\nSync Logic Verification: Passed.")
#     else:
#         print("\nSync Logic Verification: Failed")