from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from pathlib import Path
from core.processing.data_loader import load_head_data
from utils.file_management import group_files_by_person


def plot_all_scenarios_comparison(all_data, global_origin=None):
    """
    Plots a 3D comparison of head movement paths for all participants across scenario types (A, B, C).
    The function groups scenarios by type, aligns positions either to a global or the start
    point of each path, and visualizes all participants' movements in a single 3D plot per scenario type.
    Different colors represent different participants. Starting points are marked with circles.

    Parameters:
    
    :param all_data: dict 
    Nested dictionary with participant IDs as keys. Each value is a dictionary with scenario names
    as keys and Numpy arrays (NxM) of recorded positions as values. The first three columns of the array
    are assumed to be X, Y, Z coordinates.
    Example structure:
    all_data = {
        "person_1": {"1-A": np.array([...]), "1-B": np.array([...]), "1-C": np.array([...])},
        "person_2": {"2-A": np.array([...]), "2-B": np.array([...]), "2-C": np.array([...])},
    }

    :param global_origin: np.array or None
    Optional 3D point (X,Y,Z) to align all paths. If None, each scenario path is aligned to its own
    first point as original.

    Returns:
    None = Saves one 3D comparison plot per scenario type (A,B,C) in the output directory:
    'data/output/plots_comparison/<scenario_type>/comparison_<scenario_type>_scenario_3d.png'
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
        cmap = plt.cm.get_cmap("tab20", len(participant_ids))
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

            label = person_id if person_id not in plotted else None
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

        # explanation = (
        #     f"Scenario {group_type} comparison:\n"
        #     " Different colors = different participants\n"
        #     " Circles = strating points\n"
        #     " Allows comparison of movement patterns\n"
        #     "across participants for the same scenario type"
        # )

        # ax.text2D(0.02, 0.05, explanation, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.8))

        file_path = os.path.join(group_folder, f'comparison_{group_type}_scenarios_3d.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot for all {group_type} scenarios")

def plot_person_scenario_comparison(all_data, person_id, global_origin=None):
    """
    Plots a 3D comparison of all scenarios for a single participant.
    Each scenario's head movement path is plotted in a different color.
    Starting points are marked with circles. Positions can be aligned to a global origin 
    or to the starting point of each scenario.

    Parameters:
    
    :param all_data: dict
    Dictionary of scenario names as keys and corresponding position arrays as values
    for the selected participant. Each array should have at least 3 columns representing X, Y, Z coordinates.
    Example:
    all_data ={
        "1-A": np.array([...]),
        "1-B": np.array([...]),
        ...
    }

    :param person_id: str
    Identifier of the participant for whom the comparison is being plotted.

    :param global_origin: np.array or None
    Optional 3D reference point (X,Y,Z) to align all scenario paths.
    If None, each scenario is aligned to its own first point as origin.

    Returns:
    None = Saves a 3D comparison plot in the folder: 
    'data/output/person_scenarios/<person_id>/<person_id>_scenarios_comparison.png'
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
    cmap = plt.cm.get_cmap("tab20", len(scenario_names))
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
    ax.grid(True, alph=0.4)
    ax.view_init(elev=30, azim=120)

    # explanation = (
    #     "Scenario Comparison:\n"
    #     " Different colors = different scenarios\n"
    #     " Circles = starting points\n"
    #     " Allows comparison of movement patterns\n"
    #     "across different scenarios"
    # )

    # ax.text2D(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=1, bbox=dict(facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()

    output_filename = os.path.join(person_folder, f"{person_id}_scenarios_comparison.png")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot for {person_id} as {output_filename}")

def load_scenario_positions_for_persons(person_ids, json_files):
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
