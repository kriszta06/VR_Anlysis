import numpy as np
import json
import os
import pytest
import tempfile

from src.core.processing import data_loader



def make_recording(pos="(1.0, 2.0, 3.0)", rot="(0.0, 0.0, 0.0)",
                   fwd="(0.0, 0.0, 1.0)", time=0.0):
    return {
        "HeadPosition": pos,
        "HeadRotation": rot,
        "HeadForward": fwd,
        "SceneTime": time,
    }


def write_json(data: dict) -> str:
    """Write dict to a temp JSON file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def make_long_recording(duration=40.0, n_points=100):
    """
    Create a Recordings list spanning `duration` seconds with `n_points`.
    Long enough to trigger the 10-second buffer filtering.
    """
    times = np.linspace(0.0, duration, n_points)
    recordings = [
        make_recording(
            pos=f"({t:.2f}, 0.0, 0.0)",
            rot="(0.0, 0.0, 0.0)",
            fwd="(0.0, 0.0, 1.0)",
            time=float(t),
        )
        for t in times
    ]
    return recordings


class TestParseVectorString:

    def test_standard_format(self):
        result = data_loader.parse_vector_string("(1.0, 2.0, 3.0)")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_negative_values(self):
        result = data_loader.parse_vector_string("(-1.5, 0.0, 4.2)")
        np.testing.assert_array_almost_equal(result, [-1.5, 0.0, 4.2])

    def test_zeros(self):
        result = data_loader.parse_vector_string("(0.00, 0.00, 0.00)")
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_scientific_notation(self):
        result = data_loader.parse_vector_string("(1.23e-4, 5.67e2, 8.9)")
        np.testing.assert_array_almost_equal(result, [1.23e-4, 5.67e2, 8.9])

    def test_two_components_pads_with_zero(self):
        result = data_loader.parse_vector_string("(1.0, 2.0)")
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 0.0])

    def test_one_component_pads_with_zeros(self):
        result = data_loader.parse_vector_string("(5.0)")
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [5.0, 0.0, 0.0])

    def test_empty_string_returns_zeros(self):
        result = data_loader.parse_vector_string("()")
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_extra_components_ignored(self):
        # Only first three values should be used
        result = data_loader.parse_vector_string("(1.0, 2.0, 3.0, 4.0)")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_returns_numpy_array(self):
        result = data_loader.parse_vector_string("(1.0, 2.0, 3.0)")
        assert isinstance(result, np.ndarray)

    def test_output_length_is_always_three(self):
        for s in ["()", "(1.0)", "(1.0, 2.0)", "(1.0, 2.0, 3.0)", "(1.0, 2.0, 3.0, 4.0)"]:
            result = data_loader.parse_vector_string(s)
            assert len(result) == 3, f"Expected length 3 for input '{s}', got {len(result)}"



class TestLoadHeadData:


    def test_nonexistent_file_returns_empty_arrays(self):
        pos, rot, fwd, ts = data_loader.load_head_data("/nonexistent/path/file.json")
        for arr in (pos, rot, fwd, ts):
            assert isinstance(arr, np.ndarray)
            assert len(arr) == 0

    def test_invalid_json_returns_empty_arrays(self):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp.write("NOT VALID JSON {{{")
        tmp.close()
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(tmp.name)
            for arr in (pos, rot, fwd, ts):
                assert len(arr) == 0
        finally:
            os.unlink(tmp.name)

    def test_missing_recordings_key_returns_empty_arrays(self):
        path = write_json({"SomeOtherKey": []})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            for arr in (pos, rot, fwd, ts):
                assert len(arr) == 0
        finally:
            os.unlink(path)

    def test_empty_recordings_list(self):
        path = write_json({"Recordings": []})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            for arr in (pos, rot, fwd, ts):
                assert isinstance(arr, np.ndarray)
        finally:
            os.unlink(path)


    def test_short_recording_returns_unfiltered_data(self):
        """Duration <= 30s should return all records unchanged."""
        recordings = [
            make_recording(time=0.0),
            make_recording(time=10.0),
            make_recording(time=20.0),
        ]
        path = write_json({"Recordings": recordings})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            assert len(ts) == 3
            np.testing.assert_array_almost_equal(ts, [0.0, 10.0, 20.0])
        finally:
            os.unlink(path)

    def test_long_recording_filters_first_and_last_10s(self):
        """Duration > 30s should remove first and last 10 seconds."""
        recordings = make_long_recording(duration=40.0, n_points=100)
        path = write_json({"Recordings": recordings})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            assert len(ts) > 0
            assert ts[0] >= 10.0,  f"First timestamp {ts[0]:.2f} should be >= 10.0"
            assert ts[-1] <= 30.0, f"Last timestamp {ts[-1]:.2f} should be <= 30.0"
        finally:
            os.unlink(path)

    def test_long_recording_returns_fewer_records_than_total(self):
        recordings = make_long_recording(duration=40.0, n_points=100)
        path = write_json({"Recordings": recordings})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            assert len(ts) < 100
        finally:
            os.unlink(path)


    def test_output_shapes_are_consistent(self):
        recordings = make_long_recording(duration=40.0, n_points=50)
        path = write_json({"Recordings": recordings})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            n = len(ts)
            assert pos.shape == (n, 3)
            assert rot.shape == (n, 3)
            assert fwd.shape == (n, 3)
        finally:
            os.unlink(path)


    def test_missing_fields_use_defaults(self):
        """Records missing HeadPosition etc. should use default zero vectors."""
        recordings = [{"SceneTime": float(t)} for t in range(5)]
        path = write_json({"Recordings": recordings})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            assert len(ts) == 5
            np.testing.assert_array_almost_equal(pos, np.zeros((5, 3)))
        finally:
            os.unlink(path)

    def test_missing_scene_time_defaults_to_zero(self):
        recordings = [{"HeadPosition": "(1.0, 2.0, 3.0)"}]
        path = write_json({"Recordings": recordings})
        try:
            pos, rot, fwd, ts = data_loader.load_head_data(path)
            assert ts[0] == 0.0
        finally:
            os.unlink(path)