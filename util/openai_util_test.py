import unittest
from unittest.mock import patch, Mock, call

import openai

from util.openai_util import completion_with_backoff


@patch("util.openai_util.openai.Completion.create")
class TestCompletionWithBackoff(unittest.TestCase):
    def testBackOff(self, mock_create):
        # Given
        mock_create.side_effect = openai.error.RateLimitError("Fetch error")

        # When
        with self.assertRaises(Exception) as context:
            completion_with_backoff()

        # Then
        self.assertEqual(6, mock_create.call_count)
