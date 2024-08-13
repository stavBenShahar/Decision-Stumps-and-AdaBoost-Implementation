from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from sklearn.base import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for a given estimator.

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data.
    X: ndarray of shape (n_samples, n_features)
        Input data to fit.
    y: ndarray of shape (n_samples, )
        Responses of input data to fit to.
    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for the given input.
    cv: int, default=5
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds.
    validation_score: float
        Average validation score over folds.
    """
    ids = np.arange(X.shape[0])
    folds = np.array_split(ids, cv)

    scores = [(scoring(y[~np.isin(ids, fold_ids)], deepcopy(estimator).fit(X[~np.isin(ids, fold_ids)], y[~np.isin(ids, fold_ids)]).predict(X[~np.isin(ids, fold_ids)])),
               scoring(y[fold_ids], deepcopy(estimator).fit(X[~np.isin(ids, fold_ids)], y[~np.isin(ids, fold_ids)]).predict(X[fold_ids])))
              for fold_ids in folds]

    train_scores, validation_scores = zip(*scores)

    return np.mean(train_scores), np.mean(validation_scores)
