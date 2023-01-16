import unittest
from unittest.mock import patch, Mock

from stack_overflow_util import fetch_questions, fetch_answer


@patch("stack_overflow_util.time")
@patch("stack_overflow_util.requests")
class TestFetchQuestions(unittest.TestCase):
    items_mock = Mock()
    response_mock = Mock()
    response_502_mock = Mock()
    response_404_mock = Mock()

    def setUp(self):

        self.response_502_mock.status_code = 502
        self.response_404_mock.status_code = 404
        self.response_502_mock.ok = False
        self.response_404_mock.ok = False

        self.response_mock.json.return_value = {
            "items": self.items_mock
        }

    def test_happy_path(self, mock_requests, mock_time):
        # Given
        mock_requests.get.return_value = self.response_mock

        # When
        result = fetch_questions(5, 20)

        # Then
        self.assertEqual(mock_requests.get.call_count, 1, msg="Should request GET once")
        self.assertEqual(mock_requests.get.call_args[0][0], "https://api.stackexchange.com/2.2/questions", msg="URL should address the questions endpoint")
        params = mock_requests.get.call_args[1]["params"]
        self.assertEqual(params["page"], 5, msg="Should set page correctly")
        self.assertEqual(params["pagesize"], 20, msg="Should set page size correctly")
        self.assertEqual(result, self.items_mock, msg="Should return items field of the response")

    def test_exception_falls_through(self, mock_requests, mock_time):
        # Given
        expected_exception = Exception("Fetch error")
        mock_requests.get.side_effect = expected_exception

        # When
        with (self.assertRaises(Exception) as context):
            fetch_questions(3, 8)

        # Then
        self.assertEqual(context.exception, expected_exception, msg="Wrong exception raised")
        self.assertEqual(mock_requests.get.call_count, 1, msg="Should request GET once")

    def test_status_502_raises_exception(self, mock_requests, mock_time):
        # Given
        mock_requests.get.return_value = self.response_502_mock

        # When
        with (self.assertRaises(Exception) as context):
            fetch_questions(1, 10)

        # Then
        self.assertTrue("502" in str(context.exception), msg="Should throw meaningful exception")

    def test_status_502_retries_3_times_with_timeouts(self, mock_requests, mock_time):
        # Given
        mock_requests.get.return_value = self.response_502_mock

        # When
        with (self.assertRaises(Exception) as context):
            fetch_questions(1, 10)

        # Then
        self.assertEqual(4, mock_requests.get.call_count)
        self.assertEqual(4, mock_time.sleep.call_count)
        self.assertEqual([0, 1, 3, 10], [arg[0][0] for arg in mock_time.sleep.call_args_list])

    def test_status_502_retries_until_success(self, mock_requests, mock_time):
        # Given
        mock_requests.get.side_effect = [self.response_502_mock, self.response_502_mock, self.response_mock, self.response_mock]

        # When
        fetch_questions(1, 10)

        # Then
        self.assertEqual(3, mock_requests.get.call_count)
        self.assertEqual(3, mock_time.sleep.call_count)
        self.assertEqual([0, 1, 3], [arg[0][0] for arg in mock_time.sleep.call_args_list])

    def test_status_404_raises_exception(self, mock_requests, mock_time):
        # Given
        mock_requests.get.side_effect = [self.response_502_mock, self.response_404_mock]

        # When
        with (self.assertRaises(Exception) as context):
            fetch_questions(1, 10)

        # Then
        self.assertTrue("404" in str(context.exception))
        self.assertEqual(2, mock_requests.get.call_count)
        self.assertEqual(2, mock_time.sleep.call_count)
        self.assertEqual([0, 1], [arg[0][0] for arg in mock_time.sleep.call_args_list])

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