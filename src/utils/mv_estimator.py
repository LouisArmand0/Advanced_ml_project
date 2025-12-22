'''
This code is heavily borrowed from Sylvain Champonnois lecture notes from the Machine Learning
and Portfolio Management class.
'''

import pandas as pd
import numpy as np
from src.utils.metrics import sharpe_ratio
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

def compute_batch_holding(pred, V, A=None, past_h=None, constant_risk=False):
    """
    Compute Markowitz holdings with return prediction 'mu' and covariance matrix 'V'

    """
    N, _ = V.shape
    if isinstance(pred, (pd.DataFrame, pd.Series)):
       pred = pred.values
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    elif pred.shape[1] == N:
        pred = pred.T

    invV = np.linalg.inv(V)
    if A is None:
        M = invV
    else:
        U = invV.dot(A)
        if A.ndim == 1:
            M = invV -np.outer(U, U.T) / U.dot(A)
        else:
            M = invV - U.dot(np.linalg.inv(U.T.dot(A)).dot(U.T))
    h = M.dot(pred)
    if constant_risk:
        h = h/np.sqrt(np.diag(h.T.dot(V.dot(h))))
    return h.T

class MeanVariance(BaseEstimator):
    def __init__(self, transform_V=None, A=1, constant_risk=True):
        if transform_V is None:
            self.transform_V = lambda x: np.cov(x.T)
        else:
            self.transform_V = transform_V
        self.A = A
        self.constant_risk = constant_risk

    def fit(self, X, y=None):
        self.V_ = self.transform_V(y)

    def predict(self, X):
        if self.A==1:
            # Imposes that the sum of weights equals to 1, i.e long short portfolio
            T,N = X.shape
            A = np.ones(N)
        else:
            A = self.A
        h = compute_batch_holding(X, self.V_, A, constant_risk=self.constant_risk)
        return h

    def score(self, X, y):
        return sharpe_ratio(np.sum(X*y, axis=1))