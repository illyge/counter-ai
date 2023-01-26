import bentoml
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from util.train_util import create_pipeline
from util.preprocess_util import prepare_data

def pipeline():
    pipe = create_pipeline()
    pipe.set_params(**{
        'nb_col_trans__nb_pipe__c_vect__min_df': 6,
        'nb_col_trans__nb_pipe__c_vect__ngram_range': (1, 3),
        'lr__C': 10.0, 'lr__solver': 'lbfgs'})
    return pipe


def train():
    data = pd.read_json("data/preprocessed.jsonl", lines=True)

    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=30)
    selected_features = ['answer_length', 'creativity', 'stealing_strength', 'sentence_length_mean', 'vocabulary_size']

    X_train = df_full_train[['stemmed_answer'] + selected_features]
    y_train = df_full_train.target
    X_test = df_test[['stemmed_answer'] + selected_features]
    y_test = df_test.target

    classifier = pipeline()
    classifier.fit(X_train, y_train)

    print(f"Score: {f1_score(classifier.predict(X_test), y_test)}")

    print(bentoml.sklearn.save_model(
        'counter-ai-model',
        classifier
    ))


if __name__ == '__main__':
    train()