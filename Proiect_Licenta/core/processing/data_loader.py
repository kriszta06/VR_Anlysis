'''
The scope of this file is to read the data from the source and prepare it for the preprocessing pipeline.
It receives a file with the next key values:
- 'SceneTime': Time passed since the start of the scene. (in seconds)
- 'HeadPosition': (x, y, z) coordinates of the head position in the 3D space. Y is the height, X and Z are the horizontal coordinates).
- 'HeadRotation': (Pitch, Yaw, Roll) Euler angles representing the head rotation in degrees.
- 'HeadForward': (x, y, z) vector representing the direction in which the user is looking.
- 'Left/ RightHandPosition and Forward': similar to the head but for the hands.

'''

import numpy as np
np.set_printoptions(suppress=True, precision=6)
import json
import os
import re

def parse_vector_string(vector_str):
    '''
    Extracts numerical values from a the dataset. Converts a string like "(x, y, z)" into a numpy array [x, y, z].
    
    :param vector_str: A string representing a vector, typically in the format "(x, y, z)". 
    '''
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
    '''
    Reads the json file and extracts the head position, rotation, forward vector, and timestamps. 
    It looks for the 'Recordings' key in the json data, which contains a list of records. 
    Each record is expected to have 'HeadPosition', 'HeadRotation', 'HeadForward', and 'SceneTime' keys. 
    The function parses these values, converts them into numpy arrays, and returns them as separate arrays for positions, rotations, forward vectors, and timestamps.
    It pays attention to recording duration; if the total duration is too short for start/end buffering, the raw data is returned without additional trimming.

    :param file_path: Path to the json file containing the head data. The file is expected to have a specific structure with a 'Recording' key that contains a list of records, each with 'HeadPosition', 'HeadRotation', 'HeadForward', and 'SceneTime' keys.
    '''
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


# if __name__ == "__main__":

#     print("DEBUG: Starting data loading tests...")

#     # print("\n[TEST] parse_vector_string")
#     # test_vectors = [
#     #     "(1.0, 2.0, 3.0)",
#     #     "(-1.5, 0.0, 4.2)",
#     #     "(0.00, 0.00, 0.00)",
#     #     "(abc, def, ghi)", 
#     #     "(1.0, 2.0)", 
#     #     "(1.0, 2.0, 3.0, 4.0)",
#     #     "()",  # Empty
#     #     "(NaN, inf, -inf)",  # Special floats
#     #     "(1.23e-4, 5.67e2, 8.9)",  # Scientific notation
#     # ]

#     # for v in test_vectors:
#     #     print(f"DEBUG: Testing vector: {v}")
#     #     result = parse_vector_string(v)
#     #     print("---")

#     # print("\n[TEST] Mock load_head_data simulation")
#     # # Simulate load_head_data with mock data
#     # mock_data = {
#     #     "Recordings": [
#     #         {"HeadPosition": "(1.0, 2.0, 3.0)", "HeadRotation": "(10.0, 20.0, 30.0)", "HeadForward": "(0.0, 0.0, 1.0)", "SceneTime": 0.0},
#     #         {"HeadPosition": "(1.1, 2.1, 3.1)", "HeadRotation": "(11.0, 21.0, 31.0)", "HeadForward": "(0.1, 0.1, 0.9)", "SceneTime": 1.0},
#     #         {"HeadPosition": "(1.2, 2.2, 3.2)", "HeadRotation": "(12.0, 22.0, 32.0)", "HeadForward": "(0.2, 0.2, 0.8)", "SceneTime": 2.0},
#     #         # Add more for longer duration
#     #         {"HeadPosition": "(2.0, 3.0, 4.0)", "HeadRotation": "(20.0, 30.0, 40.0)", "HeadForward": "(0.0, 1.0, 0.0)", "SceneTime": 10.0},
#     #         {"HeadPosition": "(2.1, 3.1, 4.1)", "HeadRotation": "(21.0, 31.0, 41.0)", "HeadForward": "(0.1, 0.9, 0.1)", "SceneTime": 11.0},
#     #         {"HeadPosition": "(2.2, 3.2, 4.2)", "HeadRotation": "(22.0, 32.0, 42.0)", "HeadForward": "(0.2, 0.8, 0.2)", "SceneTime": 12.0},
#     #         {"HeadPosition": "(3.0, 4.0, 5.0)", "HeadRotation": "(30.0, 40.0, 50.0)", "HeadForward": "(1.0, 0.0, 0.0)", "SceneTime": 20.0},
#     #         {"HeadPosition": "(3.1, 4.1, 5.1)", "HeadRotation": "(31.0, 41.0, 51.0)", "HeadForward": "(0.9, 0.1, 0.1)", "SceneTime": 21.0},
#     #         {"HeadPosition": "(3.2, 4.2, 5.2)", "HeadRotation": "(32.0, 42.0, 52.0)", "HeadForward": "(0.8, 0.2, 0.2)", "SceneTime": 22.0},
#     #     ]
#     # }

#     # print("DEBUG: Simulating load_head_data with mock data...")
#     # # Simulate the function logic
#     # recordings = mock_data.get('Recordings', [])
#     # print(f"DEBUG: Mock recordings count: {len(recordings)}")
#     # positions = []
#     # rotations = []
#     # forward_vectors = []
#     # timestamps = []

#     # for i, recording in enumerate(recordings):
#     #     print(f"DEBUG: Processing mock recording {i+1}")
#     #     head_pos_str = recording.get('HeadPosition', '(0.00, 0.00, 0.00)')
#     #     position = parse_vector_string(head_pos_str)
#     #     positions.append(position)

#     #     head_rot_str = recording.get('HeadRotation', '(0.00, 0.00, 0.00)')
#     #     rotation = parse_vector_string(head_rot_str)
#     #     rotations.append(rotation)

#     #     head_forward_str = recording.get('HeadForward', '(0.00, 0.00, 1.00)')
#     #     forward_vector = parse_vector_string(head_forward_str)
#     #     forward_vectors.append(forward_vector)

#     #     scene_time = recording.get('SceneTime', 0.0)
#     #     timestamps.append(scene_time)

#     # positions = np.array(positions)
#     # rotations = np.array(rotations)
#     # forward_vectors = np.array(forward_vectors)
#     # timestamps = np.array(timestamps)
#     # print(f"DEBUG: Mock arrays - Positions shape: {positions.shape}, Timestamps: {timestamps}")

#     # total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
#     # print(f"DEBUG: Mock total duration: {total_duration:.2f} seconds")

#     # buffer_start_end = 10.0
#     # min_useful_data = 10.0
#     # if total_duration > (2 * buffer_start_end + min_useful_data):
#     #     start_time = timestamps[0] + buffer_start_end
#     #     end_time = timestamps[-1] - buffer_start_end
#     #     valid_indices = (timestamps >= start_time) & (timestamps <= end_time)
#     #     print(f"DEBUG: Mock filtering - Valid records: {np.sum(valid_indices)}")
#     #     positions = positions[valid_indices]
#     #     rotations = rotations[valid_indices]
#     #     forward_vectors = forward_vectors[valid_indices]
#     #     timestamps = timestamps[valid_indices]
#     # else:
#     #     print("DEBUG: Mock data too short, no filtering")

#     # print(f"DEBUG: Final mock data - Positions shape: {positions.shape}")

#     # print("\n[TEST] load_head_data with real file")

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

#     test_file = os.path.join(PROJECT_ROOT, "data", "vr_recordings", "1-A.json")

#     print(f"DEBUG: Test file path: {test_file}")
#     print(f"DEBUG: File exists: {os.path.exists(test_file)}")

#     if not os.path.exists(test_file):
#         print(f"Test file {test_file} not found.")
#     else:
#         print("DEBUG: Calling load_head_data...")
#         positions, rotations, forward_vectors, timestamps = load_head_data(test_file)
#         print("DEBUG: load_head_data completed.")
#         print(f"Final loaded data from {test_file}:")
#         print(f"Positions shape: {positions.shape}")
#         print(f"Rotations shape: {rotations.shape}")
#         print(f"Forward vectors shape: {forward_vectors.shape}")
#         print(f"Timestamps shape: {timestamps.shape}")

#         if len(timestamps) > 0:
#             print(f"First timestamp: {timestamps[0]:.2f} seconds, Last timestamp: {timestamps[-1]:.2f} seconds")

#     print("DEBUG: Data loading tests completed.")