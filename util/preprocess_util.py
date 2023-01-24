from nltk.stem import SnowballStemmer
from nltk import everygrams
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
stemmer = SnowballStemmer(language='english')


def strip_whites(df):
    """
    Removes leading and trailing whitespaces from 'human_answer' and 'ai_answer' columns of the given DataFrame.

    :param df: DataFrame to process
    :type df: pd.DataFrame
    """
    df.human_answer = df.human_answer.str.strip()
    df.ai_answer = df.ai_answer.str.strip()


def remove_duplicates(df):
    """
    Removes duplicate rows based on 'question_id' column from the given DataFrame.

    :param df: DataFrame to process
    :type df: pd.DataFrame
    """
    df.drop_duplicates(subset=['question_id'], keep='last', inplace=True)


def label(df):
    """
    Labels the DataFrame with a new column 'target' which takes the value of 0 or 1, and a new column 'answer' which is
    the 'ai_answer' if 'target' is 1, otherwise it is the 'human_answer'. Also, it drops the 'ai_answer' and 'human_answer' columns from the DataFrame.

    :param df: DataFrame to process
    :type df: pd.DataFrame
    """
    df['target'] = df.apply(lambda x: x.name % 2, axis=1)
    df['answer'] = df.apply(lambda x: x.ai_answer if x.target == 1 else x.human_answer, axis=1)
    df.drop('ai_answer', axis=1, inplace=True)
    df.drop('human_answer', axis=1, inplace=True)


def tokenize(df):
    """
    Tokenizes the 'answer' and 'question' columns of the given DataFrame using the stemmer.
    It adds the tokenized and stemmed versions of 'answer' and 'question' to the DataFrame.
    Also adds stemmed version of answer as a string joined back with spaces

    :param df: DataFrame to process
    :type df: pd.DataFrame
    """
    df['tokenized_answer'] = df.answer.apply(lambda x: [stemmer.stem(w) for w in x.split()])
    df['tokenized_question'] = df.question.apply(lambda x: [stemmer.stem(w) for w in x.split()])
    df['stemmed_answer'] = df.tokenized_answer.apply(lambda x: ' '.join(x))


def add_creativity(df):
    """
    Adds a 'creativity' column to the given DataFrame which is the ratio of new words to total words in the
    'tokenized_answer' column comparing to the 'tokenized_question' column.

    :param df: DataFrame to process
    :type df: pd.DataFrame
    """
    df['new_words'] = df.apply(
        lambda x: set([w for w in x.tokenized_answer if w not in x.tokenized_question]), axis=1)
    df['creativity'] = df.apply(
        lambda x: len(x.new_words) / len(x.tokenized_answer) if len(x.tokenized_answer) > 0 else 0, axis=1)
    df.drop('new_words', axis=1, inplace=True)


def add_vocabulary(df):
    df['n_unique_words'] = df.tokenized_answer.apply(lambda x: len(set(x)))
    df['vocabulary_size'] = df.apply(
        lambda x: x.n_unique_words / len(x.tokenized_answer) if len(x.tokenized_answer) > 0 else 0, axis=1)


def add_stealing(df):
    def find_stolen_ngrams(record):
        ngrams_question = everygrams(record.tokenized_question, min_len=2)
        ngrams_answer = everygrams(record.tokenized_answer, min_len=2)
        ngrams_set = set(ngrams_question).intersection(ngrams_answer)

        return ngrams_set

    df['stolen_ngrams'] = df.apply(find_stolen_ngrams, axis=1)
    df['stealing_strength'] = np.log1p(
        df.stolen_ngrams.apply(lambda x: max([len(ngram) for ngram in x]) if len(x) > 0 else 0))
    df['stealing_frequency'] = np.log1p(
        df.apply(lambda x: len(x.stolen_ngrams) / x.n_unique_words if x.n_unique_words > 0 else 0, axis=1))
    df.drop('stolen_ngrams', axis=1, inplace=True)


def add_answer_length(df):
    df['answer_length'] = np.log1p(df.answer.str.len())

def add_sentence_length(df):
    df['sentences'] = df.answer.apply(sent_tokenize)
    df['sentence_length_mean'] = np.log1p(
        df['sentences'].apply(lambda x: np.mean([len(s) for s in x])))
    df['sentence_length_std'] = np.log1p(df.sentences.apply(lambda x: np.std([len(s) for s in x])))
