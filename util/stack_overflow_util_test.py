import unittest
from unittest.mock import patch, Mock

from stack_overflow_util import fetch_questions_page, fetch_answer


@patch("stack_overflow_util.requests")
class TestFetchQuestions(unittest.TestCase):
    items_mock = Mock()
    response_mock = Mock()

    def setUp(self):
        self.response_mock.json.return_value = {
            "items": self.items_mock
        }

    def test_happy_path(self, mock_requests):
        # Given
        mock_requests.get.return_value = self.response_mock

        # When
        result = fetch_questions_page(5, 20)

        # Then
        self.assertEqual(mock_requests.get.call_count, 1, msg="Should request GET once")
        self.assertEqual(mock_requests.get.call_args[0][0], "https://api.stackexchange.com/2.2/questions", msg="URL should address the questions endpoint")
        params = mock_requests.get.call_args[1]["params"]
        self.assertEqual(params["page"], 5, msg="Should set page correctly")
        self.assertEqual(params["pagesize"], 20, msg="Should set page size correctly")
        self.assertEqual(result, self.items_mock, msg="Should return items field of the response")

    def test_exception_falls_through(self, mock_requests):
        # Given
        expected_exception = Exception("Fetch error")
        mock_requests.get.side_effect = expected_exception

        # When
        with (self.assertRaises(Exception) as context):
            fetch_questions_page(3, 8)

        # Then
        self.assertEqual(context.exception, expected_exception, msg="Wrong exception raised")
        self.assertEqual(mock_requests.get.call_count, 1, msg="Should request GET once")


@patch("stack_overflow_util.requests")
class TestFetchAnswer(unittest.TestCase):
    answer_mock = Mock()
    response_mock = Mock()

    def setUp(self):
        self.response_mock.json.return_value = {
            "items": [self.answer_mock]
        }

    def test_happy_path(self, mock_requests):
        # Given
        answer_id = "123"
        mock_requests.get.return_value = self.response_mock

        # When
        result = fetch_answer(answer_id)

        # Then
        self.assertEqual(mock_requests.get.call_count, 1, msg="Should request GET once")
        self.assertEqual(mock_requests.get.call_args[0][0], f"https://api.stackexchange.com/2.2/answers/{answer_id}", msg="URL should address the answers endpoint")

        params = mock_requests.get.call_args[1]["params"]
        self.assertEqual(params["site"], "stackoverflow", msg="Site param should be overflow")
        self.assertEqual(result, self.answer_mock, msg="Should return items field of the response")

    def test_exception_falls_through(self, mock_requests):
        # Given
        expected_exception = Exception("Fetch error")
        mock_requests.get.side_effect = expected_exception

        # When
        with (self.assertRaises(Exception) as context):
            fetch_answer('')

        # Then
        self.assertEqual(context.exception, expected_exception, msg="Wrong exception raised")
        self.assertEqual(mock_requests.get.call_count, 1, msg="Should request GET once")