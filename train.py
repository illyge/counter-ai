import bentoml
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from util.preprocess_util import strip_whites, remove_duplicates, label, tokenize, add_sentence_length, add_answer_length, add_creativity, add_vocabulary, add_stealing

def prepare_data(df):
    data = df.copy()

    strip_whites(data)
    remove_duplicates(data)
    label(data)
    tokenize(data)
    add_creativity(data)
    add_vocabulary(data)
    add_stealing(data)
    add_answer_length(data)
    add_sentence_length(data)

    data.fillna(0, inplace=True)

    return data

def pipeline():
    pipeline = make_preparation_pipeline(kw=True)
    steps = pipeline.steps
    steps.append(('classifier', ComplementNB()))
    return Pipeline(steps)

def train():
    data = pd.read_json("data/data.jsonl", lines=True)
    train_data = prepare_data(data)

    classifier = pipeline()
    classifier.fit(train_data, train_data.target)

    print(bentoml.sklearn.save_model(
        'twitter-disasters-model',
        classifier
    ))


if __name__ == '__main__':
    train()
