import bentoml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from util.preprocess_util import prepare_data
from util.train_util import create_pipeline


def regression_pipeline():
    pipe = create_pipeline(LogisticRegression())
    pipe.set_params(**{
        'nb_col_trans__nb_pipe__c_vect__min_df': 6,
        'nb_col_trans__nb_pipe__c_vect__ngram_range': (1, 3),
        'clf__C': 10.0, 'clf__solver': 'lbfgs', 'clf__max_iter': 500})
    return pipe


def forest_pipeline():
    pipe = create_pipeline(RandomForestClassifier())
    pipe.set_params(**{'clf__max_depth': 50, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 1,
                       'clf__min_samples_split': 10, 'clf__n_estimators': 300})
    return pipe

def xgb_pipeline():
    pipe = create_pipeline(XGBClassifier())
    pipe.set_params(**{
        'nb_col_trans__nb_pipe__c_vect__min_df': 5,
        'nb_col_trans__nb_pipe__c_vect__ngram_range': (1, 4),
        'clf__learning_rate': 0.1, 'clf__max_depth': 4, 'clf__min_child_weight': 4, 'clf__n_estimators': 100})
    return pipe

def train():
    data = pd.read_json("data/balanced_data.jsonl", lines=True)
    data = prepare_data(data, train=False)

    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    selected_features = ['creativity', 'stealing_strength', 'stealing_frequency', 'sentence_length_mean',
                         'sentence_length_std', 'answer_length', 'vocabulary_size']

    X_train = df_full_train[['stemmed_answer'] + selected_features]
    y_train = df_full_train.target
    X_test = df_test[['stemmed_answer'] + selected_features]
    y_test = df_test.target

    classifier = xgb_pipeline()
    classifier.fit(X_train, y_train)

    print(f"Score: {f1_score(classifier.predict(X_test), y_test)}")

    print(bentoml.sklearn.save_model(
        'counter-ai-model',
        classifier,
        signatures={
            "predict": {"batchable": False},
            "predict_proba": {"batchable": False},
        },
    ))


if __name__ == '__main__':
    train()
