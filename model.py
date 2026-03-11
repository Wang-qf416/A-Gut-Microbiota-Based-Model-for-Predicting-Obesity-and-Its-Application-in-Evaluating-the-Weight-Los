from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_random_forest(
    X,
    y,
    param_grid=None,
    cv_splits=5,
    random_state=42
):
    """
    Train a random forest classifier using grid search.

    Returns the best model, its parameters, and cross-validated AUC.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    param_grid : dict, optional
        Hyperparameter grid. If None, a default grid is used.
    cv_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    best_estimator : RandomForestClassifier
        Best model found by grid search.
    best_params : dict
        Best hyperparameters.
    best_score : float
        Mean cross-validated AUC of the best model.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
            "min_weight_fraction_leaf": [0.0, 0.1],
            "bootstrap": [True, False],
        }

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X, y)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )
