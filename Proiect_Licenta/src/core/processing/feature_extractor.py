from pyexpat import features
import os
from src.core.processing.data_loader import load_head_data
from src import config
import numpy as np



def extract_behavior_features(scenario_data):
    """
    Extracts a comprehensive set of behavioral features from a recorded
    three-dimensional head movement trajectory.

    The function processes a sequence of recorded head positions, rotations,
    forward direction vectors, and timestamps to compute statistical,
    kinematic, temporal, and trajectory-based descriptors. These features
    characterize movement behavior through measures such as spatial
    distribution, motion dynamics, movement variability, entropy,
    trajectory efficiency, directional consistency, and head stability.
    The resulting feature vector is intended for subsequent machine
    learning tasks, such as clustering or classification.

    Parameters
    ----------
    scenario_data : numpy.ndarray
        A two-dimensional NumPy array of shape (N, 10), where N is the number
        of recorded frames. The columns are organized as follows:

        - Columns 0–2: Head position coordinates (x, y, z).
        - Columns 3–5: Head rotation angles (pitch, yaw, roll) in degrees.
        - Columns 6–8: Head forward direction vector (x, y, z).
        - Column 9: Timestamp (SceneTime) in seconds.

    Returns
    -------
    numpy.ndarray
        A one-dimensional NumPy array containing the extracted behavioral
        features, including spatial statistics, rotational statistics,
        movement speed and acceleration metrics, entropy measures,
        trajectory efficiency, movement variability, pause frequency,
        jerk, directional asymmetry, trajectory smoothness, head stability,
        and other descriptors of motion behavior.
    """

    print("DEBUG: Starting feature extraction...")
    print(f"DEBUG: Input scenario_data shape: {scenario_data.shape}")
    
    entropy_bins = getattr(config, 'FEATURES_ENTROPY_BINS', 5)
    window = getattr(config, 'FEATURES_WINDOW_SIZE', 5)
    pause_base_threshold = getattr(config, 'FEATURES_PAUSE_BASE_THRESHOLD', 0.01)
    sharp_turn_degrees = getattr(config, 'FEATURES_SHARP_TURN_DEGREES', 30)
    autocorr_min_frames = getattr(config, 'FEATURES_AUTOCORR_MIN_FRAMES', 5)
    autocorr_max_frames = getattr(config, 'FEATURES_AUTOCORR_MAX_FRAMES', 10)

    def safe_append_or_extend(features, values):
        """
        Ensures that the returned vector has a correct structure, without errors.
        Appends a value or extends a list of values to the feature list.
        If 'values' contains a single element, it is appended.
        If 'values' contains a multiple elements, they are extended into the list.

        Parameters: 

        :param features: list => The list where extracted features are stored
        :param values: int, float or array-like => A scalar value or an array-like object containing one or more values
        """
        values = np.atleast_1d(values)
        if values.size == 1:
            features.append(values[0])
        else:
            features.extend(values)

    print("DEBUG: Extracting positions, rotations, forwards, and timestamps...")
    positions = scenario_data[:, :3]
    rotations = scenario_data[:, 3:6]
    forwards = scenario_data[:, 6:9]
    timestamps = scenario_data[:, -1]
    print(f"DEBUG: Positions shape: {positions.shape}, sample: {positions[:3] if len(positions) > 0 else 'empty'}")
    print(f"DEBUG: Rotations shape: {rotations.shape}, sample: {rotations[:3] if len(rotations) > 0 else 'empty'}")
    print(f"DEBUG: Forwards shape: {forwards.shape}, sample: {forwards[:3] if len(forwards) > 0 else 'empty'}")
    print(f"DEBUG: Timestamps shape: {timestamps.shape}, sample: {timestamps[:3] if len(timestamps) > 0 else 'empty'}")

    features = []
    print("DEBUG: Initializing features list...")

    print("DEBUG: Calculating position statistics...")
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    safe_append_or_extend(features, mean_pos)
    safe_append_or_extend(features, std_pos)
    safe_append_or_extend(features, total_distance)
    print(f"DEBUG: Mean position: {mean_pos}, Std position: {std_pos}, Total distance: {total_distance}")

    print("DEBUG: Calculating rotation statistics...")
    mean_rot = np.mean(rotations, axis=0)
    std_rot = np.std(rotations, axis=0)
    safe_append_or_extend(features, mean_rot)
    safe_append_or_extend(features, std_rot)
    print(f"DEBUG: Mean rotation: {mean_rot}, Std rotation: {std_rot}")

    print("DEBUG: Calculating forward direction statistics...")
    mean_fwd = np.mean(forwards, axis=0)
    std_fwd = np.std(forwards, axis=0)
    safe_append_or_extend(features, mean_fwd)
    safe_append_or_extend(features, std_fwd)
    print(f"DEBUG: Mean forward: {mean_fwd}, Std forward: {std_fwd}")

    print("DEBUG: Calculating time-based features...")
    total_time = np.max(timestamps) - np.min(timestamps)
    mean_delta = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
    safe_append_or_extend(features, total_time)
    safe_append_or_extend(features, mean_delta)
    print(f"DEBUG: Total time: {total_time}, Mean delta time: {mean_delta}")

    print("DEBUG: Calculating movement speeds...")
    movements = np.diff(positions, axis=0)
    movement_speeds = np.linalg.norm(movements, axis=1)
    mean_speed = np.mean(movement_speeds) if len(movement_speeds) > 0 else 0
    std_speed = np.std(movement_speeds) if len(movement_speeds) > 0 else 0
    max_speed = np.max(movement_speeds) if len(movement_speeds) > 0 else 0
    safe_append_or_extend(features, [mean_speed, std_speed, max_speed])
    print(f"DEBUG: Mean speed: {mean_speed}, Std speed: {std_speed}, Max speed: {max_speed}")
    
    print("DEBUG: Calculating acceleration...")
    if len(movement_speeds) > 1:
        acceleration = np.diff(movement_speeds)
        mean_acc = np.mean(np.abs(acceleration))
        std_acc = np.std(acceleration)
        safe_append_or_extend(features, [mean_acc, std_acc])
    else:
        safe_append_or_extend(features, [0, 0])
        mean_acc, std_acc = 0, 0
    print(f"DEBUG: Mean acceleration: {mean_acc}, Std acceleration: {std_acc}")

    print("DEBUG: Calculating movement entropy...")
    if len(movements) > 0:
        hist, _ = np.histogram(movements, bins=5)
        hist = hist.flatten()
        total = np.sum(hist)
        movement_entropy = -np.sum((hist / total) * np.log(hist / total + 1e-10)) if total > 0 else 0
    else:
        movement_entropy = 0
    safe_append_or_extend(features, movement_entropy)
    print(f"DEBUG: Movement entropy: {movement_entropy}")

    print("DEBUG: Calculating average speed over time...")
    avg_speed_time = total_distance / total_time if total_time > 0 else 0
    safe_append_or_extend(features, avg_speed_time)
    print(f"DEBUG: Average speed over time: {avg_speed_time}")

    print("DEBUG: Calculating rotation ranges and stds...")
    ptp_rot = np.ptp(rotations, axis=0)
    rotations_unwrapped = np.unwrap(np.deg2rad(rotations), axis=0)
    rotation_deltas = np.diff(rotations_unwrapped, axis=0)
    rotation_deltas_deg = np.rad2deg(rotation_deltas)
    std_rot_dynamic = np.std(rotation_deltas_deg, axis=0) if len(rotation_deltas_deg) > 0 else np.zeros(3)
    safe_append_or_extend(features, ptp_rot)
    safe_append_or_extend(features, std_rot_dynamic)
    print(f"DEBUG: Peak-to-peak rotation: {ptp_rot}, Std rotation dynamic: {std_rot_dynamic}")

    print("DEBUG: Counting sharp turns...")
    sharp_turns = 0
    if len(positions) > 2:
        for i in range(2, len(positions)):
            v1 = positions[i-1] - positions[i-2]
            v2 = positions[i] - positions[i-1]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
                if np.degrees(angle) > 30:
                    sharp_turns += 1
    safe_append_or_extend(features, sharp_turns)
    print(f"DEBUG: Sharp turns: {sharp_turns}")

    print("DEBUG: Calculating speed variability and autocorrelation...")
    if len(movement_speeds) > 1:
        cv = np.std(movement_speeds) / np.mean(movement_speeds) if np.mean(movement_speeds) > 0 else 0
        autocorr = np.corrcoef(movement_speeds[:-1], movement_speeds[1:])[0, 1] if len(movement_speeds) > 10 else 0
    else:
        cv = 0
        autocorr = 0
    safe_append_or_extend(features, [cv, autocorr])
    print(f"DEBUG: Speed CV: {cv}, Speed autocorrelation: {autocorr}")

    print("DEBUG: Calculating path efficiency...")
    direct_distance = np.linalg.norm(positions[-1] - positions[0]) if len(positions) > 1 else 0
    path_eff = direct_distance / total_distance if total_distance > 0 else 0
    safe_append_or_extend(features, path_eff)
    print(f"DEBUG: Direct distance: {direct_distance}, Path efficiency: {path_eff}")

    print("DEBUG: Calculating position variance...")
    var_pos = np.var(positions, axis=0)
    safe_append_or_extend(features, var_pos)
    print(f"DEBUG: Position variance: {var_pos}")

    print("DEBUG: Calculating unique movement directions...")
    if len(movements) > 0:
        directions = movements / (np.linalg.norm(movements, axis=1, keepdims=True) + 1e-10)
        unique_directions = np.unique(np.round(directions, 2), axis=0)
        num_unique = len(unique_directions)
    else:
        num_unique = 0
    safe_append_or_extend(features, num_unique)
    print(f"DEBUG: Unique movement directions: {num_unique}")

    
    print("DEBUG: Calculating tremor index (micro-variability of speed)")

    if len(movement_speeds) > window:
        windowed_stds = [
            np.std(movement_speeds[i:i+window])
            for i in range(len(movement_speeds) - window)
        ]
        tremor_index = np.mean(windowed_stds)
    else:
        tremor_index = np.std(movement_speeds) if len(movement_speeds) > 1 else 0
    safe_append_or_extend(features, tremor_index)
    print(f"DEBUG: Tremor index: {tremor_index:.6f}")

    print("DDEBUG: Calculating pause frequency")


    pause_threshold = max(mean_speed * 0.1, 1e-4)
    pause_frames = np.sum(movement_speeds < pause_threshold)
    pause_ratio = pause_frames / len(movement_speeds) if len(movement_speeds) > 0 else 0
    safe_append_or_extend(features, pause_ratio)
    print(f"DEBUG: Pause ratio: {pause_ratio:.4f} ({pause_frames} pause frames)")

    print("DEBUG: Calculating jerk (rate of change of acceleration)")

    if len(movement_speeds) > 2:
        jerk = np.diff(movement_speeds, n=2)
        mean_jerk = np.mean(np.abs(jerk))
        std_jerk = np.std(jerk)
    else:
        mean_jerk, std_jerk = 0, 0
    
    safe_append_or_extend(features, [mean_jerk, std_jerk])
    print(f"DEBUG: Mean jerk: {mean_jerk:.6f}, Std jerk: {std_jerk:.6f}")

    print("DEBUG: Calculating directional asymetry")

    if len(rotation_deltas) > 0:
        yaw_deltas = rotation_deltas[:, 1]
        left_turns = yaw_deltas[yaw_deltas < 0]
        right_turns = yaw_deltas[yaw_deltas > 0]
        mean_left = np.mean(np.abs(left_turns)) if len(left_turns) > 0 else 0
        mean_right = np.mean(np.abs(right_turns)) if len(right_turns) > 0 else 0
        directional_asymetry = abs(mean_left - mean_right) / (mean_left + mean_right + 1e-10)
        turn_count_asymetry = abs(len(left_turns) - len(right_turns)) / (len(yaw_deltas) + 1e-10)
    else:
        directional_asymetry = 0
        turn_count_asymetry = 0

    safe_append_or_extend(features, [directional_asymetry, turn_count_asymetry])
    print(f"DEBUG: Directional asymetry: {directional_asymetry:.4f}, Turn count asymetry: {turn_count_asymetry:.4f}")

    print("DEBUG: Calculating trajectory smoothness")

    if len(movement_speeds) > 4:
        fft_values = np.abs(np.fft.rfft(movement_speeds))
        fft_norm = fft_values / (fft_values[0] + 1e-10)

        spectral_arc_length = -np.sum(np.sqrt((1.0 / len(fft_norm))**2 + np.diff(fft_norm)**2))
    else:
        spectral_arc_length = 0
    
    safe_append_or_extend(features, [spectral_arc_length])
    print(f"DEBUG: Spectral arc length: {spectral_arc_length:.4f}")

    print("DEBUG: Calculating head stability")

    rot_delta_magnitudes = np.linalg.norm(rotation_deltas_deg, axis=1)
    if len(rot_delta_magnitudes) > window:
        rot_windowed_stds = [
            np.std(rot_delta_magnitudes[i:i+window])
            for i in range(len(rot_delta_magnitudes) - window)
        ]
        head_stability = np.mean(rot_windowed_stds)
    else:
        head_stability = np.std(rot_delta_magnitudes) if len(rot_delta_magnitudes) > 1 else 0
    
    safe_append_or_extend(features, head_stability)
    print(f"DEBUG: Head stability: {head_stability:.4f}")

    print(f"DEBUG: Total features extracted: {len(features)}")

    #print(f"DEBUG pause: threshold={pause_threshold}, pause_frames={pause_frames}, len={len(movement_speeds)}, speeds sample={movement_speeds[:5]}")
    return np.array(features)


# if __name__ == "__main__":

#     print("DEBUG : Running feature extractor script")

#     def run_debug_scenario(name, positions, rotations, forwards, timestamps):
#         print(f"\n=== DEBUG SCENARIO: {name} ===")
#         scenario_data = np.hstack([positions, rotations, forwards, timestamps.reshape(-1, 1)])
#         print(f"Scenario data shape: {scenario_data.shape}")
#         features = extract_behavior_features(scenario_data)
#         if len(feature_names) != len(features):
#             print(f"WARNING: Number of feature names ({len(feature_names)}) does not match number of features ({len(features)}).")
#         print(f"Extracted {len(features)} features:")
#         for i, (name_feat, value) in enumerate(zip(feature_names, features)):
#             print(f"{i+1:2d}. {name_feat}: {value:.4f}")

#     np.random.seed(42)
#     # num_frames = 20

#     feature_names = [
#         "mean_pos_x", "mean_pos_y", "mean_pos_z",
#         "std_pos_x", "std_pos_y", "std_pos_z",
#         "total_distance",

#         "mean_pitch", "mean_yaw", "mean_roll",
#         "std_pitch", "std_yaw", "std_roll",

#         "mean_forward_x", "mean_forward_y", "mean_forward_z",
#         "std_forward_x", "std_forward_y", "std_forward_z",

#         "total_time", "mean_delta_time",

#         "mean_speed", "std_speed", "max_speed",
#         "mean_acceleration", "std_acceleration",

#         "movement_entropy",
#         "avg_speed_time",

#         "ptp_pitch", "ptp_yaw", "ptp_roll",
#         "std_pitch_dynamic", "std_yaw_dynamic", "std_roll_dynamic",

#         "sharp_turns",
#         "speed_cv", "speed_autocorr",

#         "path_efficiency",
#         "var_pos_x", "var_pos_y", "var_pos_z",
#         "unique_movement_directions"
#     ]

#     # # Test with your positions
#     # positions_custom = np.array([
#     #     [0.04967142, -0.01382643, 0.06476885],
#     #     [0.2019744, -0.03724177, 0.04135516],
#     #     [0.35989568, 0.03950171, -0.00559228]
#     # ])
#     # rotations_custom = np.zeros((3, 3))  # Dummy rotations
#     # forwards_custom = np.tile(np.array([0, 0, 1]), (3, 1))
#     # timestamps_custom = np.linspace(0, 10, 3)
#     # run_debug_scenario("Custom Positions", positions_custom, rotations_custom, forwards_custom, timestamps_custom)

#     # # Original random scenario
#     # positions = np.cumsum(np.random.randn(num_frames, 3)*0.1, axis=0)
#     # rotations = np.random.randn(num_frames,3)*5
#     # forwards = np.tile(np.array([0, 0, 1]), (num_frames, 1))
#     # timestamps = np.linspace(0, 10, num_frames)
#     # run_debug_scenario("Random Movement", positions, rotations, forwards, timestamps)

#     # # Stationary scenario (no movement)
#     # positions_stat = np.tile(np.array([0, 0, 0]), (num_frames, 1))
#     # rotations_stat = np.zeros((num_frames, 3))
#     # forwards_stat = np.tile(np.array([0, 0, 1]), (num_frames, 1))
#     # timestamps_stat = np.linspace(0, 10, num_frames)
#     # run_debug_scenario("Stationary", positions_stat, rotations_stat, forwards_stat, timestamps_stat)

#     # # Circular movement
#     # angles = np.linspace(0, 2*np.pi, num_frames)
#     # positions_circ = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(num_frames)]) * 2
#     # rotations_circ = np.zeros((num_frames, 3))
#     # forwards_circ = np.tile(np.array([0, 0, 1]), (num_frames, 1))
#     # timestamps_circ = np.linspace(0, 10, num_frames)
#     # run_debug_scenario("Circular Movement", positions_circ, rotations_circ, forwards_circ, timestamps_circ)

#     # # High rotation scenario
#     # positions_high_rot = np.cumsum(np.random.randn(num_frames, 3)*0.05, axis=0)
#     # rotations_high_rot = np.random.randn(num_frames, 3) * 45  # Higher rotations
#     # forwards_high_rot = np.tile(np.array([0, 0, 1]), (num_frames, 1))
#     # timestamps_high_rot = np.linspace(0, 10, num_frames)
#     # run_debug_scenario("High Rotation", positions_high_rot, rotations_high_rot, forwards_high_rot, timestamps_high_rot)

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
#     test_file = os.path.join(PROJECT_ROOT, "data", "vr_recordings", "1-A.json")

#     print(f"DEBUG: test file name: {test_file}")

#     print(f"\nDEBUG: Loading test scenario from {test_file}...")
#     print(f"DEBUG: File exists: {os.path.exists(test_file)}")

#     if os.path.exists(test_file):
#         positions, rotations, forwards, timestamps = load_head_data(test_file)
#         if positions.size and rotations.size and forwards.size and timestamps.size:
#             run_debug_scenario("Scenario 1-A.json", positions, rotations, forwards, timestamps)
#         else:
#             print("WARNING: load_data returned no valid data for 1-A.json")
#     else:
#         print("ERROR: 1-A.json not found.")
        
