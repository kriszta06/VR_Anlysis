import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn  as sns
import json
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from src.evaluation.ground_truth_handler import load_ground_truth

def get_output_path():
    """
    Returns the path to the directory used for storing evaluation results,
    creating it if it does not already exist.

    The output directory is located at
    ``data/output/evaluation_results`` relative to the project root. If the
    directory is absent, it is created before its path is returned.

    Returns
    -------
    str
        Absolute path to the evaluation results directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    output_dir = os.path.join(project_root, "data", "output", "evaluation_results")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created driectory: {output_dir}")
    
    return output_dir

def calculate_and_plot_metrics(y_true, y_pred, labels=None):
    """
    Computes classification performance metrics and generates a confusion
    matrix to evaluate the agreement between predicted and ground-truth labels.

    The function calculates the overall classification accuracy and produces
    a detailed classification report. It also computes the confusion matrix,
    visualizes it as a heatmap, and saves the resulting figure to the
    evaluation results directory.

    Parameters
    ----------
    y_true : list
        Ground-truth class labels.

    y_pred : list
        Predicted class labels produced by the classification algorithm.

    labels : list, optional
        Ordered list of class labels to be used when computing the confusion
        matrix and classification report. If not provided, the labels are
        inferred from the union of the ground-truth and predicted labels.

    Returns
    -------
    None
        The function prints the evaluation metrics to the console and saves
        the confusion matrix as an image file in the evaluation results
        directory.
    """

    if not y_true:
        print("Error: No data available to calculate metrics.")
        return
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    elif all(isinstance(v, (int, np.integer)) for v in y_true) and labels == ['NONE', 'LOW', 'MEDIUM', 'HIGH']:
        labels = sorted(set(y_true) | set(y_pred))

    output_dir = get_output_path()

    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\n Overall accurcy: {accuracy:.2f}%")
    print("\n Detailed clasification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Confusion Matrix\n System Performed Evaluation', fontsize=14)
    plt.ylabel('Clinical Ground Truth (Actual)')
    plt.xlabel('Algorithm Prediction')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("Success: Confusion matrix saved as 'confusion_matrix.png'.")

def export_evaluation_summary(y_true, y_pred):
    """
    Exports a summary of the classification performance to a JSON file.

    The function computes the overall classification accuracy and stores a
    summary containing the total number of evaluated samples, the accuracy
    score, the evaluation timestamp, and the execution status. The summary
    is saved as a JSON file in the evaluation results directory.

    Parameters
    ----------
    y_true : list
        Ground-truth class labels.

    y_pred : list
        Predicted class labels produced by the classification algorithm.

    Returns
    -------
    None
        The function writes the evaluation summary to a JSON file and reports
        the execution status through console messages.
    """

    output_dir = get_output_path()

    accuracy = accuracy_score(y_true, y_pred)
    summary = {
        'total_samples': len(y_true),
        'accuracy_score': float(accuracy),
        'evaluation_timestamp': str(np.datetime64('now')),
        'status': 'COMPLETED'
    }

    json_path = os.path.join(output_dir, 'evaluation_summary.json')
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        print(f"Success: Evaluation summary exported to {json_path}")
    except Exception as e:
        print(f"Error: Failed to export summary: {e}")


def load_behavioral_classification(json_path):
    """
    Loads the behavioral classification results from a JSON file.

    The function reads the specified JSON file and extracts the participant
    classification data stored under the ``persons`` field. If the file cannot
    be accessed or parsed successfully, an empty dictionary is returned.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing the behavioral classification
        results.

    Returns
    -------
    dict
        A dictionary containing the behavioral classification results for
        each participant. If the file does not exist or cannot be read, an
        empty dictionary is returned.
    """
    if not os.path.exists(json_path):
        print(f"Error: Classification results file not found: {json_path}")
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('persons', {})
    except Exception as e:
        print(f"Error: Failed to load classification JSON: {e}")
        return {}


# if __name__ == "__main__":

#     print("Debug: Testing metrics.py with actual results for Person_1, Person_10, and Person_11")

#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     classification_path = os.path.join(base_dir, 'data', 'output', 'disability_results', 'behavioral_classification.json')
#     ground_truth_path = os.path.join(base_dir, 'data', 'ground_truth', 'ground_truth.csv')

#     selected_persons = ["Person_1", "Person_10", "Person_11"]
#     classification = load_behavioral_classification(classification_path)
#     ground_truth = load_ground_truth(ground_truth_path)

#     y_true = []
#     y_pred = []
#     missing_persons = []

#     for person_id in selected_persons:
#         if person_id not in classification:
#             missing_persons.append(person_id)
#             print(f"Warning: {person_id} missing from classification results.")
#             continue

#         if person_id not in ground_truth:
#             missing_persons.append(person_id)
#             print(f"Warning: {person_id} missing from ground truth.")
#             continue

#         try:
#             true_label = int(ground_truth[person_id])
#         except Exception as e:
#             print(f"Warning: Could not convert ground truth for {person_id}: {e}")
#             continue

#         status = classification[person_id].get('behavioral_group', 'NONE')
#         pred_label = 1 if status in ['HIGH', 'MEDIUM'] else 0

#         y_true.append(true_label)
#         y_pred.append(pred_label)
#         print(f"Added {person_id}: ground_truth={true_label}, predicted_status={status}, predicted_binary={pred_label}")

#     if y_true:
#         print(" Running metric calculations on selected persons...")
#         calculate_and_plot_metrics(y_true, y_pred, labels=[0, 1])
#         print("Testing summary export...")
#         export_evaluation_summary(y_true, y_pred)
#     else:
#         print("Error: No valid selected persons available for evaluation.")

#     if missing_persons:
#         print(f"Completed with missing persons: {sorted(set(missing_persons))}")
#     else:
#         print("Debug complete: All selected persons evaluated.")