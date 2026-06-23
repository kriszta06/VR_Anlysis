from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import os
import numpy as np
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from pathlib import Path
from core.processing.data_loader import load_head_data
from core.processing.feature_extractor import extract_behavior_features
from core.analysis.disability_engine import analyze_person_disability, compute_data_driven_thresholds
from utils.file_management import group_files_by_person
from config import disability_config as config
from scipy.spatial.distance import cdist

def create_visual_disability_report(disability_assessment):
    '''
    Generates a visual report of behavioral variation scores for all assessed persons.
    The function extracts the final disability/varition scores from the provided
    assessment dictionary, creates two bar charts (raw scores and raned classification),
    colors each bar according to the status ('HIGH', 'MEDIUM', 'LOW', 'NONE'), adds
    horizontal threshold lines, and saves the resulting figure as a PNG file
    in the 'data/output/disability_reports' directory.
    It also prints summary statistics in the console.

    Parameters:
    
    :param disability_assessmnet: dict 
    A dictionary containing the disability assessment for each person.
    Each key is a person ID and the value is a dictionary with at least the following
    keys:
    -'final_score': float -> final computed behavioral variation score
    -'status': str -> categorical status ('HIGH', 'MEDIUM', 'LOW', 'NONE')

    Returns:

    None = The function saves the report figure and a PNG file and prints summary statistics.
    No return value is provided.

    Note:
    -Saves 'behavioral_variation_report.png' in the output folder.
    -Prints the number of persons analyzed and the score range to the console.
    -Handles invalid or missing scores (NaN) by ignoring those entries.
    -Bars are colored according to the status: 'HIGH' = red, 'MEDIUM' = orange, 'LOW' = yellow, 'NONE' = green.
    -Horizontal lines indicate threshold boundaries for interpretation from config: HIGH, MEDIUM, LOW.
    '''
    try:
        persons = list(disability_assessment.keys())

        if not persons:
            print("No data for report generation")
            return
        
        variation_scores = []
        valid_persons = []

        for p in persons:
            try:
                variation_score = disability_assessment[p]['final_score']
                if not np.isnan(variation_score):
                    variation_scores.append(float(variation_score))
                    valid_persons.append(p)
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error at data extraction for {p}: {e}")
                continue
        
        if not valid_persons:
            print("No valid data for graphics generation")
            return
        
        group_colors = {
            'HIGH': 'red',
            'MEDIUM': 'orange',
            'LOW': 'yellow',
            'NONE': 'green'
        }

        if getattr(config, 'USE_DATA_DRIVEN_THRESHOLDS', False):
            thresholds = compute_data_driven_thresholds(
                variation_scores,
                getattr(config, 'DATA_DRIVEN_THRESHOLD_QUANTILES', {
                    'LOW': 0.25,
                    'MEDIUM': 0.5,
                    'HIGH': 0.75,
                })
            )
        else:
            thresholds = getattr(config, 'DISABILITY_SCORE_THRESHOLDS', {
                'HIGH': 0.72,
                'MEDIUM': 0.61,
                'LOW': 0.52
            })

        high_thr = thresholds.get('HIGH', 0.72)
        medium_thr = thresholds.get('MEDIUM', 0.61)
        low_thr = thresholds.get('LOW', 0.52)

        colors = [group_colors[disability_assessment[p]['status']] for p in valid_persons]

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'output', 'disability_reports')
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, 'behavioral_variation_report.png')

        fig = plt.figure(figsize=(16, 8))

        ax1 = plt.subplot(1, 2, 1)
        bars = ax1.bar(range(len(valid_persons)), variation_scores, color=colors, alpha=0.7)
        # ax1.set_title('Behavioral variation score', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Person')
        ax1.set_ylabel('Variation score')
        ax1.set_xticks(range(len(valid_persons)))
        ax1.set_xticklabels([p.replace('Person_', 'P') for p in valid_persons], rotation=45)
        ax1.axhline(y=high_thr, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(y=medium_thr, color='orange', linestyle='--', alpha=0.7)
        ax1.axhline(y=low_thr, color='yellow', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)

        for i, (bar, score) in enumerate(zip(bars, variation_scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
        ax2 = plt.subplot(1, 2, 2)
        sorted_indices = np.argsort(variation_scores)[::-1]
        sorted_scores = [variation_scores[i] for i in sorted_indices]
        sorted_persons = [valid_persons[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        bars = ax2.bar(range(len(sorted_persons)), sorted_scores, color=sorted_colors, alpha=0.7)
        # ax2.set_title('Classification by variation score', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Score')
        ax2.set_xticks(range(len(sorted_persons)))
        ax2.set_xticklabels([f"{i+1}. {p.replace('Person_', 'P')}" for i, p in enumerate(sorted_persons)],
                            rotation=45)
        ax2.axhline(y=high_thr, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=medium_thr, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=low_thr, color='yellow', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)

        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # plt.suptitle('Behavioral Variation Report', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Report saved as: {file_path}")
        print(f" {len(valid_persons)} persons")
        print(f"Variation scores: {min(variation_scores):.3f} - {max(variation_scores):.3f}")

    except Exception as e:
        print(f"Error at report generation: {e}")
        import traceback
        traceback.print_exc()


def plot_person_dendrogram(all_person_features, person_ids, disability_assessment=None):
    """
    Generates a hierarchical clustering dendrogram showing behavioral similarity
    between participants based on their feature vectors.
    The function performs the following steps:
    -Extracts feature vectors for the specified participants.
    -Standardizes the features using z-score normalization.
    -Performs hierarchial clustering using Ward's linkage method.
    -Plots and saves a dendrogram with participant IDs labeled on the x-axis.
    -Annotates the figure with an explanation of how to interpret the dendrogram

    Parameters:
    
    :param all_person_features: dict
    A dictionary where keys are participant IDs (str) and values are
    numpy arrays representing the participant's combined feature vector.

    :param person_ids: list of str
    A list of participnt IDs to include in the dendrogram.
   
    Returns:

    str or None:
    The file path to the saved dendrogram PNG if successful, or None if 
    the dendrogra could not be generated due to empty data or errors.

    Note:
    -Creates the folder 'data/output/dendrograms' if it does not exist.
    -Saves the dendrogram figure as 'participant_similarity_dendrogram.png'
    -Prints status messages and error messages to the console.
    -Lower vertical connections indicate participants with more similar behavior
    -Height of teh horizontal lines corresponds to the behavioral distance.
    -Clusters visually indicate groups of participants with similar navigation patterns.
    """
    STATUS_COLORS = {
        'HIGH': 'red',
        'MEDIUM': 'orange',
        'LOW': 'yellow',
        'NONE': 'green'
    }

    try:
        if not person_ids:
            print("No participants provided for dendrogram")
            return None

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'output', 'dendrograms')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'participant_similarity_dendrogram.png')

        if len(person_ids) == 1:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, person_ids[0], ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()

            print("Only one participant available. Saved a simple participant label instead of a dendrogram.")
            return file_path
        
        feature_matrix = np.array([all_person_features[pid] for pid in person_ids])

        if feature_matrix.size == 0:
            print("Feature matrix is empty")
            return None
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        z = linkage(scaled_features, method='ward')

        centroid = np.mean(scaled_features, axis=0)
        dist_to_centroid = np.linalg.norm(scaled_features - centroid, axis=1)
        pairwise_mean_dist = np.mean(cdist(scaled_features, scaled_features), axis=1)
        std_dist = np.std(dist_to_centroid) if np.std(dist_to_centroid) > 0 else 1.0
        std_cons = np.std(pairwise_mean_dist) if np.std(pairwise_mean_dist) > 0 else 1.0
        anomaly_scores = (
            (dist_to_centroid - np.mean(dist_to_centroid)) / std_dist +
            (pairwise_mean_dist - np.mean(pairwise_mean_dist)) / std_cons
        ) / 2.0
        outlier_threshold = np.percentile(anomaly_scores, 75)
        outlier_flags = {
            pid: score >= outlier_threshold 
            for pid, score in zip(person_ids, anomaly_scores)
        }

        has_scores = (
            disability_assessment is not None and
            all(pid in disability_assessment for pid in person_ids)
        )

        if has_scores:
            score_matrix = np.array([
                [
                    disability_assessment[pid]['distance_score'],
                    disability_assessment[pid]['mahalanobis_score'],
                    disability_assessment[pid]['consistency_score'],
                    disability_assessment[pid]['final_score']
                ]

                for pid in person_ids
            ])

            score_scaler = StandardScaler()
            scaled_scores = score_scaler.fit_transform(score_matrix)
            z_scores = linkage(scaled_scores, method='ward')
        
        fig, axes = plt.subplots(1, 2, figsize=(26, 11))
        fig.subplots_adjust(wspace=0.12)

        ax_left = axes[0]
        plt.sca(ax_left)
        dendrogram(
            z,
            labels=person_ids,
            orientation='top',
            leaf_rotation=90,
            leaf_font_size=9,
            show_leaf_counts=True,
            above_threshold_color='#9ecae1',
            ax=ax_left,
        )

        for tick in ax_left.get_xticklabels():
            pid = tick.get_text()
            if outlier_flags.get(pid, False):
                tick.set_color('#d62728')
                tick.set_fontweight('bold')

        # ax_left.set_title('Behavioral Similarity Dendrogram', fontsize=13, fontweight='bold', pad=12)

        ax_left.set_xlabel('Participant ID', fontsize=11)
        ax_left.set_ylabel('Behavioral Distance', fontsize=11)
        ax_left.grid(True, alpha=0.25, linestyle='--')

        ax_right = axes[1]

        if has_scores:
            plt.sca(ax_right)
            dendrogram(
                z_scores,
                labels=person_ids,
                orientation='top',
                leaf_rotation=90,
                leaf_font_size=9,
                show_leaf_counts=True,
                above_threshold_color='#fdae6b',
                ax=ax_right,
            )

            for tick in ax_right.get_xticklabels():
                pid = tick.get_text()
                status = disability_assessment.get(pid, {}).get('status', 'NONE')
                tick.set_color(STATUS_COLORS.get(status, 'black'))
                if status in ('HIGH', 'MEDIUM'):
                    tick.set_fontweight('bold')

            # ax_right.set_title('Behavioral Variation Classification', fontsize=13, fontweight='bold', pad=12)
            
            ax_right.set_xlabel('Participant ID', fontsize=11)
            ax_right.set_ylabel('Behavioral Distance', fontsize=11)
            ax_right.grid(True, alpha=0.25, linestyle='--')

            
        else:
            ax_right.text(
                0.5, 0.5,
                "Disability assessment not available",
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax_right.transAxes
            )
            ax_right.set_title("Behavioral Variation Classification", fontsize=13, fontweight='bold', pad=12)
            ax_right.axis('off')
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'output', 'dendrograms')
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, 'participant_similarity_dendrogram.png')


        plt.figure(figsize=(18, 10))
        dendrogram(
            z,
            labels=person_ids,
            orientation='top',
            leaf_rotation=90,
            leaf_font_size=10,
            show_leaf_counts=True,
            above_threshold_color='#bcbddc')

        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        print("Dendrogram saved as: participant_similarity_dendrogram.png")
    
        return file_path
    
    except Exception as e:
        print(f"Error generating dendrogram: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def create_disability_report(disability_assessment):
    """
    Generates a detailed behavioral variaton report for all participants,
    saves it both as a text file and a PDF, and prints it to the console.
    The report includes:
    -Individual participant scores and cluster assignments
    -Behavioral variation category (HIGH, MEDIUM, LOW, NONE)
    -Overall statistics across participants
    -Distribution of participants per cluster

    Parameters:

    :param disability_assessment: dict
    Dictionary with participant IDs as keys and assessment results as values.
    Each value is expected to contain at least the following keys:
    -'final_score': float, overall behavioral variation score
    -'status': str, variation category ('HIGH', 'MEDIUM', 'LOW', 'NONE')
    -'cluster': int, cluster ID assigned by clustering
    -'distance_score': float, distance-based score
    -'mahalanobis_score': float, Mahalanobis distance-based score
    -'consistency_score': float, score reflecting behavioral consistency

    Returns:

    dict
    Paths to the saved reports:
    -'txt': path to the TXT report
    -'pdf': path to the PDF report
    Returns None if no data is provided.
    """

    if not disability_assessment:
        print("No data available for report.")
        return None
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'output', 'text_reports')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = os.path.join(output_dir, f"behavioral_report_{timestamp}.txt")
    pdf_path = os.path.join(output_dir, f"behavioral_report_{timestamp}.pdf")

    report_lines = []
    report_lines.append("="*100)
    report_lines.append("Behavioral Variation Report")
    report_lines.append("="*100)

    categories = {
        "HIGH": [],
        "MEDIUM": [],
        "LOW": [],
        "NONE": []
    }

    for person, result in disability_assessment.items():
        categories[result['status']].append((person, result))
        
    for status, person in categories.items():
        if person:
            report_lines.append(f"\nParticipants with {status.lower()} variation ({len(person)}):")
            report_lines.append("-"*70)

            for person, result in sorted(person, key=lambda x: x[1]['final_score'], reverse=True):
                report_lines.append(f"{person}:")
                report_lines.append(f" Final score: {result['final_score']:.3f} | Cluster: {result['cluster']}")
                report_lines.append(f" Distance score: {result['distance_score']}")
                report_lines.append(f" Mahalanobis score: {result['mahalanobis_score']:.3f}")
                report_lines.append(f" Consistency score: {result['consistency_score']:.3f}")
                report_lines.append("")
    
    total_persons = len(disability_assessment)
    high_count = len(categories["HIGH"])
    medium_count = len(categories["MEDIUM"])
    low_count = len(categories["LOW"])
    minimal_count = len(categories["NONE"])

    persons_with_variation = high_count + medium_count + low_count
    percentage_with_variation = (persons_with_variation / total_persons)*100
    report_lines.append("\n Overall Statistics:")
    report_lines.append("-"*50)
    report_lines.append(f" Total participants analized: {total_persons}")
    report_lines.append(f" Participants with hight variation: {high_count} ({high_count/total_persons*100:.1f}%)")
    report_lines.append(f" Participants with medium variation: {medium_count} ({medium_count/total_persons*100:.1f}%)")
    report_lines.append(f" Participants with low variation: {low_count} ({low_count/total_persons*100:.1f}%)")
    report_lines.append(f" Participants with minimal variation: {minimal_count} ({minimal_count/total_persons*100:.1f}%)")
    report_lines.append(f" Total number of participants with behavioral variation: {persons_with_variation} ({percentage_with_variation:.1f}%)")

    clusters = {}
    for person, result in disability_assessment.items():
        clusters.setdefault(result['cluster'], []).append(person)
    
    report_lines.append(f"\n Cluster Distribution:")
    report_lines.append("="*30)

    for cluster_id, persons in clusters.items():
        cluster_statuses = [disability_assessment[p]['status'] for p in persons]

        report_lines.append(f" Cluster {cluster_id}: {len(persons)} participants")
        report_lines.append(
            f" HIGH: {cluster_statuses.count('HIGH')} | "
            f"MEDIUM: {cluster_statuses.count('MEDIUM')} | "
            f"LOW: {cluster_statuses.count('LOW')} | "
            f"NONE: {cluster_statuses.count('NONE')}"
        )
    
    for line in report_lines:
        print(line)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    elements = []

    for line in report_lines:
        if line.strip():
            elements.append(Paragraph(line, styles["Normal"]))
        else:
            elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)

    print(f"\nReport saved as TXT: {txt_path}")
    print(f"Report saved as PDF: {pdf_path}")

    return {
        "txt": txt_path,
        "pdf": pdf_path
    }
    

# if __name__ == "__main__":

#     print("\nDebug: Running reports.py")

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
#     vr_dir = Path(PROJECT_ROOT) / "data" / "vr_recordings"

#     print(f"DEBUG: VR directory: {vr_dir}")
    
#     json_files = sorted(vr_dir.glob("*.json"))
#     grouped = group_files_by_person(json_files)

#     selected_persons = sorted(grouped.keys())[:3]
#     grouped = {person: grouped[person] for person in selected_persons}

#     print(f"Total JSON files: {len(json_files)}")
#     print(f"Total persons: {len(grouped)} (first 3 selected)")
#     print(f"Selected persons: {selected_persons}")

#     person_aggregated_data = {}

#     for person, files in grouped.items():
#         print(f"\nProcessing {person}")
#         person_features = []
#         total_scenarios = len(files)

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

#     if not person_aggregated_data:
#         print("\nNo valid person data available for report generation.")
#     else:
#         disability_assessment = analyze_person_disability(person_aggregated_data)

#         print(f"\nDisability assessment generated for {len(disability_assessment)} persons")
#         for person, data in disability_assessment.items():
#             print(f" {person}: final_score={data['final_score']:.3f}, status={data['status']}, cluster={data['cluster']}")

#         report_paths = create_disability_report(disability_assessment)
#         create_visual_disability_report(disability_assessment)

#         if report_paths:
#             print(f"Report saved successfully")
#             print(f" TXT: {report_paths['txt']}")
#             print(f" PDF: {report_paths['pdf']}")
#         else:
#             print("Report generation failed")

#         person_ids = list(person_aggregated_data.keys())
#         if len(person_ids) >= 2:
#             print(f"\nGenerating dendrogram for {len(person_ids)} participants")
#             dendrogram_path = plot_person_dendrogram(
#                 all_person_features=person_aggregated_data,
#                 person_ids=person_ids
#             )
#             if dendrogram_path:
#                 print(f"Dendrogram saved at: {dendrogram_path}")
#             else:
#                 print("Dendrogram generation failed")
#         else:
#             print("Skipping dendrogram: need at least 2 valid persons")

#     print("\nDebug: Reports workflow completed")
