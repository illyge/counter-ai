from util.preprocess_util import strip_whites, remove_duplicates, label, tokenize, \
    add_creativity, add_vocabulary, add_stealing, add_answer_length, add_sentence_length

import pandas as pd
import unittest
from unittest.mock import patch
import numpy as np


class TestStripWhites(unittest.TestCase):
    def test_strip_whites(self):
        # Given
        df = pd.DataFrame({
            'human_answer': ['  hello  ', '\nworld  '],
            'ai_answer': ['  how  ', '  are      '],
        })

        # When
        strip_whites(df)

        # Then
        self.assertEqual(df.human_answer.tolist(), ['hello', 'world'])
        self.assertEqual(df.ai_answer.tolist(), ['how', 'are'])

    def test_strip_whites_with_empty_string(self):
        # Given
        df = pd.DataFrame({
            'human_answer': [' ', '  '],
            'ai_answer': ['   ', '    '],
        })

        # When
        strip_whites(df)

        # Then
        self.assertEqual(df.human_answer.tolist(), ['', ''])
        self.assertEqual(df.ai_answer.tolist(), ['', ''])

    def test_strip_whites_with_null_values(self):
        # Given
        df = pd.DataFrame({
            'human_answer': ['  hello  ', '  world  ', None],
            'ai_answer': ['  how  ', '  are  ', None],
        })

        # When
        strip_whites(df)

        # Then
        self.assertEqual(df.human_answer.tolist(), ['hello', 'world', None])
        self.assertEqual(df.ai_answer.tolist(), ['how', 'are', None])


class TestRemoveDuplicates(unittest.TestCase):
    def test_remove_duplicates(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4, 4],
            'human_answer': ['hello', 'world', 'how', 'are', 'are'],
            'ai_answer': ['hi', 'planet', 'what', 'you', 'you'],
        })

        # When
        remove_duplicates(df)

        # Then
        self.assertEqual(df.question_id.tolist(), [1, 2, 3, 4])
        self.assertEqual(df.human_answer.tolist(), ['hello', 'world', 'how', 'are'])
        self.assertEqual(df.ai_answer.tolist(), ['hi', 'planet', 'what', 'you'])

    def test_remove_duplicates_with_empty_dataframe(self):
        # Given
        df = pd.DataFrame()

        # When
        remove_duplicates(df)

        # Then
        self.assertTrue(df.empty)

    def test_remove_duplicates_with_no_duplicates(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'human_answer': ['hello', 'world', 'how', 'are'],
            'ai_answer': ['hi', 'planet', 'what', 'you'],
        })

        # When
        remove_duplicates(df)

        # Then
        self.assertEqual(df.question_id.tolist(), [1, 2, 3, 4])
        self.assertEqual(df.human_answer.tolist(), ['hello', 'world', 'how', 'are'])
        self.assertEqual(df.ai_answer.tolist(), ['hi', 'planet', 'what', 'you'])

    def test_remove_duplicates_with_duplicates_on_other_column(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4, 4],
            'human_answer': ['hi', 'world', 'how', 'are', 'are'],
            'ai_answer': ['hi', 'planet', 'what', 'you', 'you'],
        })

        # When
        remove_duplicates(df)

        # Then
        self.assertEqual(df.question_id.tolist(), [1, 2, 3, 4])
        self.assertEqual(df.human_answer.tolist(), ['hi', 'world', 'how', 'are'])
        self.assertEqual(df.ai_answer.tolist(), ['hi', 'planet', 'what', 'you'])


class TestLabel(unittest.TestCase):
    def test_label(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'human_answer': ['hello', 'world', 'how', 'are'],
            'ai_answer': ['hi', 'planet', 'what', 'you'],
        })

        # When
        label(df)

        # Then
        self.assertEqual(df.target.tolist(), [0, 1, 0, 1])
        self.assertEqual(df.answer.tolist(), ['hello', 'planet', 'how', 'you'])
        self.assertNotIn('human_answer', df.columns)
        self.assertNotIn('ai_answer', df.columns)


class TestTokenize(unittest.TestCase):

    @patch('util.preprocess_util.stemmer')
    def test_tokenize(self, mock_stemmer):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2],
            'question': ['first question', 'second one?'],
            'answer': ['the answer', 'Another answer'],
        })
        mock_stemmer.stem.side_effect = lambda x: x + '_stemmed'

        # When
        tokenize(df)

        # Then
        self.assertEqual(df.tokenized_answer.tolist(),
                         [['the_stemmed', 'answer_stemmed'], ['Another_stemmed', 'answer_stemmed']])
        self.assertEqual(df.tokenized_question.tolist(),
                         [['first_stemmed', 'question_stemmed'], ['second_stemmed', 'one?_stemmed']])
        self.assertEqual(df.stemmed_answer.tolist(),
                         ['the_stemmed answer_stemmed', 'Another_stemmed answer_stemmed'])


class TestAddCreativity(unittest.TestCase):
    def test_add_creativity(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'tokenized_question': [['What', 'is'], ['What', 'is'], ['How', 'old'], ['What', 'is']],
            'tokenized_answer': [['Paris', 'is'], ['Jupiter', 'is', 'big'], ['4.5', 'billion', 'years'],
                                 ['Giraffe', 'is', 'tall']],
        })

        # When
        add_creativity(df)

        # Then
        np.testing.assert_almost_equal([0.5, 0.66, 1.0, 0.66], df.creativity.tolist(), 2)
        self.assertNotIn('new_words', df.columns)

    def test_add_creativity_with_empty_tokenized_answer(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'tokenized_question': [['What', 'is'], ['What', 'is'], ['How', 'old'], ['What', 'is']],
            'tokenized_answer': [[], [], [], []],
        })

        # When
        add_creativity(df)

        # Then
        self.assertEqual(df.creativity.tolist(), [0, 0, 0, 0])
        self.assertNotIn('new_words', df.columns)


class TestAddVocabulary(unittest.TestCase):
    def test_add_vocabulary(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'tokenized_answer': [['Paris', 'is', 'Paris'], ['Jupiter', 'is', 'big'], ['4.5', '4.5', '4.5'],
                                 ['Giraffe']],
        })

        # When
        add_vocabulary(df)

        # Then
        np.testing.assert_almost_equal([0.66, 1.0, 0.33, 1.0], df.vocabulary_size.tolist(), 2)
        self.assertEqual([2, 3, 1, 1], df.n_unique_words.tolist())

    def test_add_vocabulary_with_empty_tokenized_answer(self):
        # Given
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'tokenized_answer': [[], [], [], []],
        })

        # When
        add_vocabulary(df)

        # Then
        self.assertEqual([0, 0, 0, 0], df.vocabulary_size.tolist())
        self.assertEqual([0, 0, 0, 0], df.n_unique_words.tolist())


class TestAddStealing(unittest.TestCase):
    @patch('util.preprocess_util.np.log1p')
    def test_add_stealing(self, mock_log1p):
        # Given
        mock_log1p.side_effect = lambda x: x
        df = pd.DataFrame({
            'question_id': [1, 2, 3, 4],
            'tokenized_question': [['how', 'long', 'is', 'stolen_ngram'],
                                   ['what', 'about', 'two', 'words', 'long'],
                                   ['and', 'three', 'words', 'ngram'],
                                   ['empty_answer']],
            'tokenized_answer': [['stolen_ngram_', 'is', 'one_word', 'long'],
                                 ['two', 'words', 'ngram'],
                                 ['three', 'words', 'ngram', 'contains_two_words_ngrams'],
                                 []],
            'n_unique_words': [4, 3, 4, 0]
        })
        # When
        add_stealing(df)

        # Then
        np.testing.assert_almost_equal([0.0, 2, 3, 0], df.stealing_strength.tolist(), 2)
        np.testing.assert_almost_equal([0.0, 0.33, 0.75, 0], df.stealing_frequency.tolist(), 2)
        self.assertNotIn('stolen_ngrams', df.columns)

class TestAddAnswerLength(unittest.TestCase):
    @patch('util.preprocess_util.np.log1p')
    def test_add_answer_length(self, mock_log1p):
        # Given
        mock_log1p.side_effect = lambda x: x
        df = pd.DataFrame({'answer': ['This is a test', 'This is another test', '']})

        # When
        add_answer_length(df)

        # Then
        self.assertEqual([14, 20, 0], df.answer_length.tolist())
        # np.testing.assert_almost_equal([14, 20, 0], df.answer_length.tolist(), 2)
        self.assertIn('answer_length', df.columns)

class TestAddSentenceLength(unittest.TestCase):
    @patch('util.preprocess_util.np.log1p')
    def test_add_sentence_length(self, mock_log1p):
        # Given
        labeled_data = pd.DataFrame({
            'answer': [
                'This is a test. This is another test.',
                'This is a short answer. The quick brown fox jumps over the lazy dog.',
                'The sky is blue. The sun is yellow. The grass is green.'
            ]
        })
        mock_log1p.side_effect = lambda x: x
        # When
        add_sentence_length(labeled_data)

        # Then
        np.testing.assert_almost_equal([18.0, 33.5, 17.67], labeled_data.sentence_length_mean.tolist(), 2)
        np.testing.assert_almost_equal([3.0, 10.5, 1.25], labeled_data.sentence_length_std.tolist(), 2)
        self.assertIn('sentences', labeled_data.columns)
        self.assertIn('sentence_length_mean', labeled_data.columns)
        self.assertIn('sentence_length_std', labeled_data.columns)