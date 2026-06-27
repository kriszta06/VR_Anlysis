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
from src.core.processing.data_loader import load_head_data
from src.core.processing.feature_extractor import extract_behavior_features
from src.core.analysis.disability_engine import analyze_person_disability, compute_data_driven_thresholds
from src.utils.file_management import group_files_by_person
from src.config import disability_config as config
from scipy.spatial.distance import cdist

def create_visual_disability_report(disability_assessment):
    """
    Generates a visual summary of the behavioral variation assessment for all
    evaluated participants.

    The function extracts the final behavioral variation scores and their
    corresponding categorical labels from the assessment results, generates two
    bar charts (individual scores and ranked scores), applies color coding based
    on the assigned behavioral variation category, overlays the decision
    thresholds used for interpretation, and saves the resulting figure as a
    high-resolution image. Summary statistics are also printed to the console.

    Parameters
    ----------
    disability_assessment : dict
        Dictionary containing the behavioral variation assessment for each
        participant. Each entry is expected to contain at least:

        - ``final_score`` : float
            Final behavioral variation score.
        - ``status`` : str
            Behavioral variation category (e.g., ``"HIGH"``, ``"MEDIUM"``,
            ``"LOW"``, or ``"NONE"``).

    Returns
    -------
    None
        The function does not return a value. Instead, it generates and saves a
        visual report and prints summary statistics to the console.

    Outputs
    -------
    The following file is generated:

    - ``behavioral_variation_report.png``

    stored in:

    ``data/output/disability_reports/``

    The report contains:

    - A bar chart displaying the behavioral variation score of each participant.
    - A ranked bar chart showing participants ordered by decreasing score.

    Notes
    -----
    - Participants with invalid or missing scores (e.g., NaN) are excluded from
    the visualizations.
    - Errors encountered while extracting individual participant data are handled
    gracefully, allowing report generation to continue.
    - Score thresholds are obtained either from the project configuration or are
    computed dynamically from the data, depending on the configuration
    settings.
    - Bar colors correspond to the assigned behavioral variation category:
    ``HIGH`` (red), ``MEDIUM`` (orange), ``LOW`` (yellow), and ``NONE``
    (green).
    - Summary statistics, including the number of analyzed participants and the
    range of behavioral variation scores, are printed to the console.
    """
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
    Generates hierarchical clustering visualizations illustrating behavioral
    similarity between participants based on their extracted feature vectors and,
    optionally, their behavioral variation assessment scores.

    The function standardizes participant feature vectors, performs hierarchical
    clustering using Ward's linkage method, and generates a dendrogram
    representing behavioral similarity. When behavioral assessment results are
    available, a second dendrogram is generated using the assessment scores
    instead of the original feature vectors. Both visualizations are saved as
    high-resolution images.

    Parameters
    ----------
    all_person_features : dict[str, numpy.ndarray]
        Dictionary mapping participant identifiers to their corresponding
        combined feature vectors.

    person_ids : list[str]
        List of participant identifiers to include in the clustering analysis.

    disability_assessment : dict, optional
        Dictionary containing the behavioral variation assessment for each
        participant. When provided, it is expected to contain the following
        values for every participant:

        - ``distance_score``
        - ``mahalanobis_score``
        - ``consistency_score``
        - ``final_score``
        - ``status``

        If all requested participants are present, an additional dendrogram
        based on these scores is generated.

    Returns
    -------
    dict or None
        Dictionary containing the paths of the generated visualizations:

        - ``features`` : str
            Path to the dendrogram generated from the participant feature
            vectors.
        - ``scores`` : str or None
            Path to the dendrogram generated from the behavioral assessment
            scores, or ``None`` if it could not be generated.

        Returns ``None`` if the visualization cannot be produced due to invalid
        input data or an unexpected error.

    Outputs
    -------
    The generated figures are stored in:

    ``data/output/dendrograms/``

    and include:

    - ``dendrogram_features.png``
    - ``dendrogram_scores.png`` (if behavioral assessment data are available)

    Notes
    -----
    - Feature vectors are standardized using z-score normalization prior to
    clustering.
    - Hierarchical clustering is performed using Ward's linkage method.
    - If only one participant is provided, a simple figure containing the
    participant label is generated instead of a dendrogram.
    - In the feature-based dendrogram, participants identified as behavioral
    outliers are highlighted.
    - In the score-based dendrogram, participant labels are colored according to
    their behavioral variation category.
    - Errors encountered during processing are reported, and the function
    returns ``None`` if the dendrogram generation fails.
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

        path_features = os.path.join(output_dir, 'dendrogram_features.png')
        path_scores = os.path.join(output_dir, 'dendrogram_scores.png')

        if len(person_ids) == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, person_ids[0], ha='center', va='center', fontsize=16)
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(path_features, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("Only one participant available. Saved a simple participant label instead of a dendrogram.")
            return {'features': path_features, 'scores': None}
        
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
        outlier_mean = np.mean(anomaly_scores)
        outlier_std = np.std(anomaly_scores) if np.std(anomaly_scores) > 0 else 1.0
        outlier_threshold = outlier_mean + 1.0 * outlier_std
        outlier_flags = {
            pid: score >= outlier_threshold 
            for pid, score in zip(person_ids, anomaly_scores)
        }

        fig1, ax1 = plt.subplots(figsize=(14, 8))
        max_y = 30
        dendrogram(
            z,
            labels=person_ids,
            orientation='top',
            leaf_rotation=90,
            leaf_font_size=9,
            show_leaf_counts=True,
            above_threshold_color='#9ecae1',
            ax=ax1,
        )

        ax1.set_ylim(0, max_y)

        for tick in ax1.get_xticklabels():
            pid = tick.get_text()
            if outlier_flags.get(pid, False):
                tick.set_color('red')
                tick.set_fontweight('bold')

        ax1.set_xlabel('Participants ID', fontsize=11)
        ax1.set_ylabel('Behavioral Distance', fontsize=11)
        ax1.grid(True, alpha=0.25, linestyle='--')
        fig1.tight_layout()
        fig1.savefig(path_features, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"Dendrogram (features) saved: {path_features}")

        saved_scores_path = None

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
        
            z_scores_scaled = z_scores.copy()
            z_scores_scaled[:, 2] = (z_scores[:,2] / np.max(z_scores[:,2])) ** 0.5 * np.max(z[:,2])

            fig2, ax2 = plt.subplots(figsize=(14, 8))

            dendrogram(
                z_scores_scaled,
                labels=person_ids,
                orientation='top',
                leaf_rotation=90,
                leaf_font_size=9,
                show_leaf_counts=True,
                above_threshold_color='#9ecae1',
                ax=ax2
            )
            
            ax2.set_ylim(0, max_y)

            for tick in ax2.get_xticklabels():
                pid = tick.get_text()
                status = disability_assessment.get(pid, {}).get('status', 'NONE')
                tick.set_color(STATUS_COLORS.get(status, 'black'))
                if status in ('HIGH', 'MEDIUM'):
                    tick.set_fontweight('bold')
            
            ax2.set_xlabel('Participants ID', fontsize=11)
            ax2.set_ylabel('Behavioral Distance', fontsize=11)
            ax2.grid(True, alpha=0.25, linestyle='--')
            fig2.tight_layout()
            fig2.savefig(path_scores, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            saved_scores_path = path_scores
            print(f"Dendrogram (scores) saved: {path_scores}")
        
        else:
            print("Dendrogram (scores) not available")

        return {'features': path_features, 'scores': saved_scores_path}
    
        
    except Exception as e:
        print(f"Error generating dendrogram: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def create_disability_report(disability_assessment):
    """
    Generates a detailed behavioral variation report for all evaluated
    participants and exports it in both plain text and PDF formats.

    The report summarizes the behavioral variation assessment for each
    participant, including the computed scores, assigned behavioral variation
    category, and cluster membership. It also provides aggregate statistics and
    the distribution of participants across behavioral variation categories and
    clusters. The generated report is printed to the console and saved as both
    TXT and PDF files.

    Parameters
    ----------
    disability_assessment : dict
        Dictionary mapping participant identifiers to their behavioral
        assessment results. Each participant entry is expected to contain at
        least the following fields:

        - ``final_score`` : float
            Overall behavioral variation score.
        - ``status`` : str
            Behavioral variation category (``"HIGH"``, ``"MEDIUM"``,
            ``"LOW"``, or ``"NONE"``).
        - ``cluster`` : int
            Identifier of the cluster assigned to the participant.
        - ``distance_score`` : float
            Distance-based behavioral score.
        - ``mahalanobis_score`` : float
            Mahalanobis distance-based behavioral score.
        - ``consistency_score`` : float
            Behavioral consistency score.

    Returns
    -------
    dict or None
        Dictionary containing the paths of the generated reports:

        - ``txt`` : str
            Path to the generated text report.
        - ``pdf`` : str
            Path to the generated PDF report.

        Returns ``None`` if the input assessment dictionary is empty.

    Outputs
    -------
    The following files are generated in:

    ``data/output/text_reports/``

    - ``behavioral_report_<timestamp>.txt``
    - ``behavioral_report_<timestamp>.pdf``

    Notes
    -----
    - Reports are timestamped to avoid overwriting previous results.
    - The report includes participant-level behavioral assessment, overall
    summary statistics, and the distribution of participants across clusters
    and behavioral variation categories.
    - The complete report is also printed to the console.
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
