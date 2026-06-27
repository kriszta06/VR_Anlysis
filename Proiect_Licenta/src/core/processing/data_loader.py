import numpy as np
np.set_printoptions(suppress=True, precision=6)
import json
import os
import re

def parse_vector_string(vector_str):
    """
    Parses a string representation of a three-dimensional vector and converts
    it into a NumPy array.

    The function extracts all numerical values from the input string using a
    regular expression. If fewer than three numerical values are identified,
    the missing components are padded with zeros to ensure a three-dimensional
    output vector.

    Parameters
    ----------
    vector_str : str
        A string containing the vector components, typically in the format
        "(x, y, z)".

    Returns
    -------
    numpy.ndarray
        A one-dimensional NumPy array of length three containing the parsed
        floating-point values.
    """
    # print(f"DEBUG parse_vector_string: Input string: '{vector_str}'")

    numbers = re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', vector_str)
    # print(f"DEBUG parse_vector_string: Extracted numbers: {numbers}")

    while len(numbers) < 3:
        numbers.append('0.0')
    result = np.array([float(numbers[0]), float(numbers[1]), float(numbers[2])])
    # print(f"DEBUG parse_vector_string: Parsed vector: {result}")
    # print(f"Result: {result}")
    return result

def load_head_data(file_path):
    """
    Loads head-tracking data from a JSON file and extracts the recorded head
    position, rotation, forward direction, and corresponding timestamps.

    The function reads the JSON file, retrieves the list of recordings stored
    under the ``Recordings`` key, and converts the recorded vector quantities
    into NumPy arrays. If the recording duration is sufficient, the first and
    last 10 seconds are discarded to remove potential initialization and
    termination artifacts. For recordings with insufficient duration, the raw
    data are returned without temporal filtering.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing the recorded head-tracking data.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple containing:
        - head positions,
        - head rotations,
        - head forward vectors,
        - timestamps.

        If the file cannot be loaded or does not contain a valid
        ``Recordings`` field, four empty NumPy arrays are returned.
    """
    print(f"DEBUG load_head_data: Starting to load data from {file_path}")
    try: 
        print("DEBUG load_head_data: Attempting to open and load JSON file")
        with open(file_path, 'r') as file:
            data = json.load(file)
        print("DEBUG load_head_data: JSON loaded successfully")
    except Exception as e:
        print(f"DEBUG load_head_data: Error loading JSON: {e}")
        print("DEBUG load_head_data: Returning empty arrays")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    if 'Recordings' not in data:
        print("DEBUG load_head_data: 'Recordings' key not found in data")
        print("DEBUG load_head_data: Returning empty arrays")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    recordings = data.get('Recordings', [])
    # print(f"DEBUG load_head_data: Found {len(recordings)} recordings")
    positions = []
    rotations = []
    forward_vectors = []
    timestamps = []

    print("DEBUG load_head_data: Parsing each recording...")
    for i, recording in enumerate(recordings):
        # print(f"DEBUG load_head_data: Processing recording {i+1}/{len(recordings)}")
        head_pos_str = recording.get('HeadPosition', '(0.00, 0.00, 0.00)')
        # print(f"DEBUG load_head_data: HeadPosition string: '{head_pos_str}'")
        position = parse_vector_string(head_pos_str)
        positions.append(position)

        head_rot_str = recording.get('HeadRotation', '(0.00, 0.00, 0.00)')
        # print(f"DEBUG load_head_data: HeadRotation string: '{head_rot_str}'")
        rotation = parse_vector_string(head_rot_str)
        rotations.append(rotation)

        head_forward_str = recording.get('HeadForward', '(0.00, 0.00, 1.00)') 
        # print(f"DEBUG load_head_data: HeadForward string: '{head_forward_str}'")
        forward_vector = parse_vector_string(head_forward_str)
        forward_vectors.append(forward_vector)

        scene_time = recording.get('SceneTime', 0.0)
        # print(f"DEBUG load_head_data: SceneTime: {scene_time}")
        timestamps.append(scene_time)

    # print("DEBUG load_head_data: Converting lists to NumPy arrays")
    positions = np.array(positions)
    rotations = np.array(rotations)
    forward_vectors = np.array(forward_vectors)
    timestamps = np.array(timestamps)
    # print(f"DEBUG load_head_data: Arrays created - Positions shape: {positions.shape}, Rotations shape: {rotations.shape}, Forwards shape: {forward_vectors.shape}, Timestamps shape: {timestamps.shape}")

    if len(timestamps) > 0:
        total_duration = timestamps[-1] - timestamps[0] 
        file_name = os.path.basename(file_path)
        # print(f"DEBUG - Loaded {file_name}: {len(timestamps)} records, total duration: {total_duration:.2f} seconds")
    else:
        file_name = os.path.basename(file_path)
        print(f"DEBUG - Loaded {file_name}: No valid records found.")

    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
    # print(f"DEBUG load_head_data: Total duration calculated: {total_duration:.2f} seconds")

    buffer_start_end = 10.0
    min_useful_data = 10.0
    # print(f"DEBUG load_head_data: Buffer settings - start/end: {buffer_start_end}s, min useful: {min_useful_data}s")

    if total_duration <= (2 * buffer_start_end + min_useful_data):
        file_name = os.path.basename(file_path)
        print(f"WARNING - {file_name} has insufficient data for processing (total duration: {total_duration:.2f} seconds). Skipping filtering.")
        print("DEBUG load_head_data: Returning unfiltered data")
        return positions, rotations, forward_vectors, timestamps
    
    start_time = timestamps[0] + buffer_start_end
    end_time = timestamps[-1] - buffer_start_end
    # print(f"DEBUG load_head_data: Filtering - start_time: {start_time:.2f}, end_time: {end_time:.2f}")
    valid_indices = (timestamps >= start_time) & (timestamps <= end_time)
    # print(f"DEBUG load_head_data: Valid indices count: {np.sum(valid_indices)} out of {len(timestamps)}")

    useful_duration = timestamps[valid_indices][-1] - timestamps[valid_indices][0] 
    print(f"SUCCESS - Data loaded and filtered: {len(timestamps)} total records, {len(timestamps[valid_indices])} valid records, useful duration: {useful_duration:.2f} seconds.")

    print("DEBUG load_head_data: Returning filtered data")
    return positions[valid_indices], rotations[valid_indices], forward_vectors[valid_indices], timestamps[valid_indices]


if __name__ == "__main__":

    print("DEBUG: Starting data loading tests...")


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    test_file = os.path.join(PROJECT_ROOT, "data", "vr_recordings", "1-A.json")

    print(f"DEBUG: Test file path: {test_file}")
    print(f"DEBUG: File exists: {os.path.exists(test_file)}")

    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found.")
    else:
        print("DEBUG: Calling load_head_data...")
        positions, rotations, forward_vectors, timestamps = load_head_data(test_file)
        print("DEBUG: load_head_data completed.")
        print(f"Final loaded data from {test_file}:")
        print(f"Positions shape: {positions.shape}")
        print(f"Rotations shape: {rotations.shape}")
        print(f"Forward vectors shape: {forward_vectors.shape}")
        print(f"Timestamps shape: {timestamps.shape}")

        if len(timestamps) > 0:
            print(f"First timestamp: {timestamps[0]:.2f} seconds, Last timestamp: {timestamps[-1]:.2f} seconds")

    print("DEBUG: Data loading tests completed.")