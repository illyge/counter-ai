import unittest

from download_questions import download_questions
from unittest.mock import Mock, patch


@patch("download_questions.fetch_questions")
@patch("download_questions.jsonlines.open")
@patch("download_questions.time.time")
class TestDownloadQuestions(unittest.TestCase):
    q1_mock = None
    q2_mock = None
    file_mock = None
    mock_questions = None

    def setUp(self):
        self.mock_questions = [Mock() for i in range(0, 5)]
        self.file_mock = Mock()

    def test_happy_path(self, mock_time, jsonl_mock_open, mock_fetch_questions):
        # Given
        fetch_return_values = [
            self.mock_questions[:3],
            self.mock_questions[3:]
        ]
        mock_fetch_questions.side_effect = fetch_return_values
        n_pages = len(fetch_return_values)
        mock_time.return_value = 3452346.3245346
        jsonl_mock_open.return_value.__enter__.return_value = self.file_mock

        # When
        download_questions(range(1, n_pages+1))

        # Then
        self.assertEqual(1, jsonl_mock_open.call_count, msg="Open file should be called once")
        self.assertEqual(2, mock_fetch_questions.call_count, msg="Fetch questions should be called for each page")
        self.assertEqual("./data/raw/questions/3452346.jsonl", jsonl_mock_open.call_args[0][0],
                         msg="File name should be the current timestamp as int number of seconds and the path should be correct")
        self.assertEqual("w", jsonl_mock_open.call_args[0][1],
                         msg="File should be open for writing")

        self.assertEqual(2, self.file_mock.write_all.call_count, msg="There should be a file write for each page")
        self.assertEqual(3, len(self.file_mock.write_all.call_args_list[0].args[0]))
        self.assertEqual(2, len(self.file_mock.write_all.call_args_list[1].args[0]))



class PG(unittest.TestCase):
    def test_d(self):
        a = [i for i in range(0, 5)]
        print (a[3:])
