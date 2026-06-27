from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from pathlib import Path
from src.core.processing.data_loader import load_head_data
from src.utils.file_management import group_files_by_person


def plot_all_scenarios_comparison(all_data, global_origin=None):
    """
    Generates comparative three-dimensional visualizations of participant
    trajectories for each scenario type.

    The function groups scenarios according to their type (e.g., A, B, or C),
    normalizes the recorded trajectories relative to either a common global
    origin or the initial position of each trajectory, and overlays the
    trajectories of all participants in a single 3D visualization for each
    scenario type. Each participant is assigned a unique color to facilitate
    visual comparison.

    Parameters
    ----------
    all_data : dict
        Nested dictionary mapping participant identifiers to their corresponding
        scenario data. Each participant entry is expected to contain a
        dictionary whose keys are scenario names and whose values contain the
        recorded positional data. The first three columns of each data array
        represent the three-dimensional coordinates (x, y, z).

        Example:

            {
                "Person_1": {
                    "1-A": np.ndarray,
                    "1-B": np.ndarray,
                    "1-C": np.ndarray
                },
                "Person_2": {
                    "2-A": np.ndarray,
                    "2-B": np.ndarray,
                    "2-C": np.ndarray
                }
            }

    global_origin : numpy.ndarray, optional
        Three-dimensional reference point used to normalize all trajectories to
        a common coordinate system. If ``None``, each trajectory is normalized
        relative to its initial position.

    Returns
    -------
    None
        The function does not return a value. Instead, it generates and saves
        one comparison plot for each available scenario type.

    Outputs
    -------
    The generated figures are stored in:

    ``data/output/plots_comparison/<scenario_type>/``

    using the filename:

    - ``comparison_<scenario_type>_scenarios_3d.png``

    Notes
    -----
    - Separate visualizations are generated for each scenario type (A, B, and
    C), provided that corresponding data are available.
    - Each participant is represented by a unique color across all trajectories
    within the same scenario type.
    - The starting point of every trajectory is highlighted.
    - Trajectories are normalized before visualization to improve
    comparability between participants.
    - The viewing angle and axis orientation are configured to match the
    coordinate conventions used throughout the project.
    """
    if not all_data:
        print("No data available for plotting")
        return
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'output', 'plots_comparison')
    os.makedirs(output_dir, exist_ok=True)

    scenario_groups = {'A':[], 'B':[], 'C':[]}

    for person_id, scenarios_data in all_data.items():
        for scenario_name, data in scenarios_data.items():
            group_type = scenario_name.split('-')[1] if '-' in scenario_name else scenario_name
            if group_type in scenario_groups:
                scenario_groups[group_type].append({
                    'person_id': person_id,
                    'scenario_name': scenario_name,
                    'data': data
                })
    
    for group_type, scenarios in scenario_groups.items():
        if not scenarios:
            continue

        group_folder = os.path.join(output_dir, group_type)
        os.makedirs(group_folder, exist_ok=True)

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_title(f'Comarison of all {group_type} Scenarios\n(3D Head Movement Routes)', fontsize=16, fontweight='bold')

        participant_ids = list(set(s['person_id'] for s in scenarios))
        cmap = plt.get_cmap("tab20", len(participant_ids))
        colors = [cmap(i) for i in range(len(participant_ids))]
        color_map = {pid: colors[i] for i, pid in enumerate(participant_ids)}

        plotted = set()

        for scenario_info in scenarios:
            person_id = scenario_info['person_id']
            scenario_name = scenario_info['scenario_name']
            data = scenario_info['data']

            positions = data[:, :3]
            if global_origin is not None:
                positions = positions - global_origin
            else:
                positions = positions - positions[0]

            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]

            plotted.add(person_id)

            ax.plot(x, y, z, color=color_map[person_id],
                linewidth=2.0,
                alpha=0.85,
                label=person_id if scenario_name.endswith(group_type) else None)
            
            ax.scatter(x[0], y[0], z[0],
                       marker='o', color=color_map[person_id], s=100, edgecolors='black')
        
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Z Position", fontsize=12)
        ax.set_zlabel("Y Position", fontsize=12)

        ax.grid(True, alpha=0.4)
        ax.view_init(elev=30, azim=120)
        ax.legend(fontsize=10, loc='lower right')

        file_path = os.path.join(group_folder, f'comparison_{group_type}_scenarios_3d.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot for all {group_type} scenarios")

def plot_person_scenario_comparison(all_data, person_id, global_origin=None):
    """
    Generates a three-dimensional comparative visualization of all scenario
    trajectories associated with a single participant.

    The function overlays the recorded trajectories from all available
    scenarios for the specified participant in a single 3D plot. Each scenario
    is represented using a distinct color, allowing visual comparison of
    movement patterns across scenarios. Trajectories may be aligned to a common
    global reference point or normalized relative to their individual starting
    positions.

    Parameters
    ----------
    all_data : dict
        Dictionary mapping scenario names to their corresponding positional
        data arrays. The first three columns of each array are assumed to
        represent the three-dimensional coordinates (x, y, z).

        Example::

            {
                "1-A": np.ndarray,
                "1-B": np.ndarray,
                "1-C": np.ndarray
            }

    person_id : str
        Identifier of the participant whose scenario trajectories are being
        visualized.

    global_origin : numpy.ndarray, optional
        Three-dimensional reference point used to normalize all trajectories to
        a common coordinate system. If ``None``, each trajectory is normalized
        relative to its initial position.

    Returns
    -------
    None
        The function does not return a value. Instead, it generates and saves a
        three-dimensional comparison plot.

    Outputs
    -------
    The generated figure is stored in:

    ``data/output/person_scenarios/<person_id>/``

    using the filename:

    - ``<person_id>_scenarios_comparison.png``

    Notes
    -----
    - Each scenario is displayed using a unique color.
    - The starting position of every trajectory is highlighted.
    - Trajectories are normalized before visualization to improve
    comparability between scenarios.
    - The output directory is created automatically if it does not already
    exist.
    - The viewing angle and axis orientation are configured to match the
    coordinate conventions used throughout the project.
    """

    if not all_data:
        print("No data available for plotting")
        return
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, 'data', 'output', 'person_scenarios')
    os.makedirs(output_base, exist_ok=True)

    person_folder = os.path.join(output_base, person_id)
    os.makedirs(person_folder, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title(f'Comparison of All Scenarios for {person_id}\n(3D Head Movement Routes)', fontsize=16, fontweight='bold')
    
    scenario_names = list(all_data.keys())
    cmap = plt.get_cmap("tab20", len(scenario_names))
    colors = [cmap(i) for i in range(len(scenario_names))]

    for idx, (scenario_name, data) in enumerate(all_data.items()):
        color = colors[idx]

        positions = data[:, :3]

        if global_origin is not None:
            positions = positions - global_origin
        else:
            positions = positions - positions[0]
        
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        ax.plot(x, y, z,
                color=color,
                linewidth=2.5,
                alpha=0.8,
                label=scenario_name)
        
        ax.scatter(x[0], y[0], z[0],
                   marker='o', color=color, s=120, edgecolors='black')
    
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Z Position", fontsize=12)
    ax.set_zlabel("Y Position", fontsize=12)
    ax.legend(loc='best', fontsize=11)

    ax.grid(True, alpha=0.4)
    ax.view_init(elev=30, azim=120)

    plt.tight_layout()

    output_filename = os.path.join(person_folder, f"{person_id}_scenarios_comparison.png")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot for {person_id} as {output_filename}")

def load_scenario_positions_for_persons(person_ids, json_files):

    """
    Loads and validates the positional data associated with the requested
    participants.

    The function groups the available JSON files by participant, loads the head
    position data for each scenario, performs basic validation, and returns the
    valid trajectories organized by participant. Scenarios containing no valid
    samples or an excessive proportion of padded positions are excluded from the
    output.

    Parameters
    ----------
    person_ids : list[str]
        List of participant identifiers whose scenario data should be loaded.

    json_files : list[pathlib.Path]
        List of paths to the available JSON recording files.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Nested dictionary mapping each participant identifier to a dictionary of
        valid scenarios. Each scenario is represented by a NumPy array
        containing the corresponding three-dimensional positional data.

        Example::

            {
                "Person_1": {
                    "1-A": np.ndarray,
                    "1-B": np.ndarray
                },
                "Person_2": {
                    "2-A": np.ndarray
                }
            }

    Notes
    -----
    - JSON files are grouped by participant using
    ``group_files_by_person()``.
    - Scenarios with no valid positional samples are skipped.
    - If the positional data contain at least three coordinates, the axes are
    reordered from ``(X, Y, Z)`` to ``(X, Z, Y)`` to match the coordinate
    convention used throughout the project.
    - Scenarios for which more than 50% of the recorded positions correspond to
    zero-valued padding are excluded.
    - Participants for whom no valid scenarios are available are omitted from
    the returned dictionary.
    - Informational and warning messages are printed during the loading
    process.
    """
    all_data = {}
    grouped = group_files_by_person(json_files)

    for person in person_ids:
        if person not in grouped:
            print(f"Warning: {person} not found in VR recordings")
            continue

        person_data = {}
        for file_path in sorted(grouped[person], key=lambda p: p.name):
            print(f"Loading {file_path.name} for {person}")
            positions, _, _, timestamps = load_head_data(file_path)

            if len(timestamps) == 0 or positions.size == 0:
                print(f"  Skipping {file_path.name}: no valid head data")
                continue

            if positions.shape[1] >= 3:
                positions = positions[:, [0, 2, 1]]

            zero_ratio = np.all(positions == 0, axis=1).mean()
            if zero_ratio > 0.5:
                print(f"  Skipping {file_path.name}: too much padding ({zero_ratio:.2f})")
                continue

            person_data[file_path.stem] = positions
            print(f"  Loaded scenario {file_path.stem}: {positions.shape}")

        if person_data:
            all_data[person] = person_data
        else:
            print(f"No valid scenarios loaded for {person}")

    return all_data


# if __name__ == "__main__":
#     base_dir = Path(__file__).resolve().parent.parent
#     recordings_dir = base_dir / "data" / "vr_recordings"

#     print("\nDebug: Running scenario comparison workflow from actual VR recordings")

#     if not recordings_dir.exists():
#         raise FileNotFoundError(f"VR recordings directory not found: {recordings_dir}")

#     json_files = sorted(recordings_dir.glob("*.json"))
#     if not json_files:
#         raise FileNotFoundError(f"No JSON recordings found in: {recordings_dir}")

#     grouped_files = group_files_by_person(json_files)
#     selected_persons = sorted(grouped_files.keys())[:3]

#     print(f"Selected persons for verification: {selected_persons}")

#     all_data = load_scenario_positions_for_persons(selected_persons, json_files)

#     if not all_data:
#         print("No valid data found for the selected persons. Exiting.")
#     else:
#         print("\nDebug: Generating individual scenario comparison plots for selected persons")
#         for person_id, person_scenarios in all_data.items():
#             print(f"  Plotting scenarios for {person_id}")
#             plot_person_scenario_comparison(person_scenarios, person_id)

#         print("\nDebug: Generating global scenario comparison across selected persons")
#         plot_all_scenarios_comparison(all_data)

#         print("\nDebug: Scenario comparison workflow completed successfully")
