from sklearn.model_selection import GridSearchCV
import warnings


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
