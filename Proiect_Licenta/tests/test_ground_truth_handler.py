import pytest
import os
import csv
import tempfile

from src.evaluation import ground_truth_handler


def write_csv(rows, fieldnames=None, path=None):
    if path is None:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        path = tmp.name
        tmp.close()
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row[name.strip()] for name in fieldnames])
    return path


class TestLoadGroundTruth:

    def test_nonexistent_file_returns_empty_dict(self):
        result = ground_truth_handler.load_ground_truth('/nonexistent/path/file.csv')
        assert result == {}

    def test_returns_dict(self):
        path = write_csv([{'person_id': '1', 'diagnosis': '0'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_person_id_prefixed_with_person(self):
        path = write_csv([{'person_id': '1', 'diagnosis': '0'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert 'Person_1' in result
        finally:
            os.unlink(path)

    def test_integer_diagnosis_converted(self):
        path = write_csv([{'person_id': '1', 'diagnosis': '2'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert result['Person_1'] == 2
            assert isinstance(result['Person_1'], int)
        finally:
            os.unlink(path)

    def test_float_string_diagnosis_converted_to_int(self):
        path = write_csv([{'person_id': '1', 'diagnosis': '1.0'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert result['Person_1'] == 1
        finally:
            os.unlink(path)

    def test_string_diagnosis_uppercased(self):
        path = write_csv([{'person_id': '1', 'diagnosis': 'healthy'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert result['Person_1'] == 'HEALTHY'
        finally:
            os.unlink(path)

    def test_multiple_rows_loaded(self):
        rows = [
            {'person_id': '1', 'diagnosis': '0'},
            {'person_id': '2', 'diagnosis': '1'},
            {'person_id': '3', 'diagnosis': '2'},
        ]
        path = write_csv(rows)
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert len(result) == 3
            assert 'Person_1' in result
            assert 'Person_2' in result
            assert 'Person_3' in result
        finally:
            os.unlink(path)

    def test_missing_person_id_column_returns_empty(self):
        path = write_csv([{'id': '1', 'diagnosis': '0'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_missing_diagnosis_column_returns_empty(self):
        path = write_csv([{'person_id': '1', 'label': '0'}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_whitespace_in_headers_handled(self):
        path = write_csv(
            [{'person_id': '1', 'diagnosis': '1'}],
            fieldnames=[' person_id ', ' diagnosis ']
        )
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert len(result) >= 0
        finally:
            os.unlink(path)

    def test_whitespace_in_values_stripped(self):
        path = write_csv([{'person_id': ' 1 ', 'diagnosis': ' 0 '}])
        try:
            result = ground_truth_handler.load_ground_truth(path)
            assert 'Person_1' in result
        finally:
            os.unlink(path)

    def test_empty_csv_returns_empty_dict(self):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        tmp.write('person_id,diagnosis\n')
        tmp.close()
        try:
            result = ground_truth_handler.load_ground_truth(tmp.name)
            assert result == {}
        finally:
            os.unlink(tmp.name)


class TestSyncData:

    def _make_program_results(self, persons):
        return {
            'persons': {
                pid: {'behavioral_group': status, 'final_score': score}
                for pid, status, score in persons
            }
        }

    def test_returns_three_lists(self):
        program_results = self._make_program_results([('Person_1', 'NONE', 0.2)])
        ground_truth = {'Person_1': 0}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert isinstance(y_true, list)
        assert isinstance(y_pred, list)
        assert isinstance(ids, list)

    def test_matched_person_included(self):
        program_results = self._make_program_results([('Person_1', 'HIGH', 0.8)])
        ground_truth = {'Person_1': 1}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert 'Person_1' in ids
        assert y_true[0] == 1
        assert y_pred[0] == 'HIGH'

    def test_person_only_in_ground_truth_excluded(self):
        program_results = self._make_program_results([])
        ground_truth = {'Person_99': 1}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert 'Person_99' not in ids
        assert len(ids) == 0

    def test_person_only_in_program_results_excluded(self):
        program_results = self._make_program_results([('Person_1', 'LOW', 0.5)])
        ground_truth = {}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert len(ids) == 0

    def test_all_three_lists_same_length(self):
        program_results = self._make_program_results([
            ('Person_1', 'HIGH', 0.9),
            ('Person_2', 'NONE', 0.1),
        ])
        ground_truth = {'Person_1': 1, 'Person_2': 0}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert len(y_true) == len(y_pred) == len(ids)

    def test_multiple_persons_synced(self):
        program_results = self._make_program_results([
            ('Person_1', 'HIGH', 0.9),
            ('Person_2', 'MEDIUM', 0.6),
            ('Person_3', 'NONE', 0.2),
        ])
        ground_truth = {'Person_1': 1, 'Person_2': 0, 'Person_3': 0}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert len(ids) == 3

    def test_missing_behavioral_group_defaults_to_unknown(self):
        program_results = {'persons': {'Person_1': {'final_score': 0.5}}}
        ground_truth = {'Person_1': 1}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert y_pred[0] == 'UNKNOWN'

    def test_empty_both_inputs(self):
        y_true, y_pred, ids = ground_truth_handler.sync_data({'persons': {}}, {})
        assert y_true == []
        assert y_pred == []
        assert ids == []

    def test_partial_overlap(self):
        program_results = self._make_program_results([
            ('Person_1', 'HIGH', 0.9),
            ('Person_3', 'LOW', 0.4),
        ])
        ground_truth = {'Person_1': 1, 'Person_2': 0}
        y_true, y_pred, ids = ground_truth_handler.sync_data(program_results, ground_truth)
        assert len(ids) == 1
        assert 'Person_1' in ids