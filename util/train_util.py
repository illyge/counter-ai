import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings

from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


def grid_search(classifier, param_grid, X, y, cv=5):
    """
    grid_search(classifier, param_grid, X, y)

    Performs a grid search on a given classifier with specified parameters.

    Parameters:
    - classifier: The classifier to perform the grid search on.
    - param_grid (dict): The parameter grid to search over.
    - X (array-like): The input data.
    - y (array-like): The target data.

    Returns:
    - gs (GridSearchCV): The grid search object.
    """
    gs = GridSearchCV(classifier,
                      param_grid=param_grid,
                      scoring='f1',
                      cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    print("Tuned Hyperparameters:", gs.best_params_)
    print(f"F1 score : {gs.best_score_}")
    return gs


class ComplementNBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clf = ComplementNB()

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        probas = self.clf.predict_proba(X)
        return pd.DataFrame(np.array([x[1] for x in probas]), columns=['nb_probability'])


def create_pipeline():
    nb_pipe = Pipeline([('c_vect', CountVectorizer(min_df=7, ngram_range=(1, 3))),
                        ('nb_proba', ComplementNBTransformer())])

    columns = [('nb_pipe', nb_pipe, 'stemmed_answer')]

    col_trans = ColumnTransformer(columns,
                                  remainder='passthrough')  # the `remainder` part ensures that `selected_features` are passed down the pipeline and later combined with the output of nb_pipe

    return Pipeline([('nb_col_trans', col_trans),
                     ('lr', LogisticRegression(C=10.0, solver='lbfgs', max_iter=500))])
