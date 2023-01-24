import unittest
from unittest.mock import patch, Mock
import pandas as pd

from train import prepare_data


class TestPrepareData(unittest.TestCase):
    @patch('train.add_sentence_length')
    @patch('train.add_answer_length')
    @patch('train.add_stealing')
    @patch('train.add_vocabulary')
    @patch('train.add_creativity')
    @patch('train.tokenize')
    @patch('train.label')
    @patch('train.remove_duplicates')
    @patch('train.strip_whites')
    def test_prepare_data(self, mock_strip_whites, mock_remove_duplicates, mock_label, mock_tokenize,
                          mock_add_creativity, mock_add_vocabulary, mock_add_stealing, mock_add_answer_length, mock_add_sentence_length):
        # Given

        df = Mock()
        data = df.copy.return_value

        called_mocks = []

        mock_strip_whites.side_effect = called_mocks.append(mock_strip_whites)
        mock_remove_duplicates.side_effect = called_mocks.append(mock_remove_duplicates)
        mock_label.side_effect = called_mocks.append(mock_label)
        mock_tokenize.side_effect = called_mocks.append(mock_tokenize)

        mock_add_creativity.side_effect = called_mocks.append(mock_add_creativity)
        mock_add_vocabulary.side_effect = called_mocks.append(mock_add_vocabulary)
        mock_add_stealing.side_effect = called_mocks.append(mock_add_stealing)
        mock_add_answer_length.side_effect = called_mocks.append(mock_add_answer_length)
        mock_add_sentence_length.side_effect = called_mocks.append(mock_add_sentence_length)

        # When
        result = prepare_data(df)

        # Then
        # Assert that all the functions were called with the object
        expected_mocks = [mock_strip_whites, mock_remove_duplicates, mock_label, mock_tokenize,
                          mock_add_creativity, mock_add_vocabulary, mock_add_stealing, mock_add_answer_length, mock_add_sentence_length]

        for expected in expected_mocks:
            expected.assert_called_once_with(data)

        # Assert that the function calls are in the correct order
        self.assertEqual(expected_mocks, called_mocks)

        # Assert that Nones are filled with 0s
        data.fillna.assert_called_once_with(0, inplace=True)

        # Assert that returns data
        self.assertEqual(result, data)