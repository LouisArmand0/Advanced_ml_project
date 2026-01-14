from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils.mv_estimator import MeanVariance
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.metaestimators import _safe_split

def compute_pnl(h, ret,  pred_lag):
    pnl = h.shift(pred_lag).mul(ret)
    if isinstance(h, pd.DataFrame):
        pnl = pnl.sum(axis=1)
    return pnl

def fit_predict(estimator, X, y, train, test, return_estimator=True):
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    if return_estimator:
        return estimator.predict(X_test), estimator
    else:
        return estimator.predict(X_test)

@dataclass
class Backtester:
    estimator: BaseEstimator = MeanVariance()
    max_train_size: int = 36
    min_train_size: int = 5
    test_size: int = 1
    pred_lag: int = 1
    start_date: str = '2011-12-30'
    end_date: str = None
    name: str = None

    def compute_holdings(self, X, y, pre_dispatch="2*n_jobs", n_jobs=1):

        # determine n_splits considering min_train_size
        n_obs = len(X.loc[self.start_date: self.end_date])
        max_train = self.max_train_size
        min_train = self.min_train_size

        # TimeSeriesSplit requires n_splits >=1
        n_splits = max((n_obs - min_train) // self.test_size, 1)

        cv = TimeSeriesSplit(
            max_train_size=max_train,
            test_size=self.test_size,
            n_splits=n_splits,
        )
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        res = parallel(
            delayed(fit_predict)(
                clone(self.estimator), X.values, y.values, train, test, True
            )
            for train, test in cv.split(X)
        )
        y_pred, estimators = zip(*res)
        idx = X.index[np.concatenate([test for _, test in cv.split(X)])]
        if isinstance(y, pd.DataFrame):
            cols = y.columns
            h = pd.DataFrame(np.concatenate(y_pred), index=idx, columns=cols)
        elif isinstance(y, pd.Series):
            h = pd.Series(np.concatenate(y_pred), index=idx)
        else:
            h = None
        self.h_ = h
        self.estimators_ = estimators
        self.cv_ = cv
        return self

    def compute_pnl(self, ret):
        pnl = compute_pnl(self.h_, ret, self.pred_lag)

        self.pnl_ = pnl.loc[self.start_date: self.end_date]
        if self.name:
            self.pnl_ = self.pnl_.rename(self.name)
        return self

    def train(self, X, y, ret):
        self.compute_holdings(X, y)
        self.compute_pnl(ret)
        return self.pnl_


### BACKTEST ENGINE FOR GNN ###
def yearly_walkforward_splits(dates, train_years=1, test_years=1):
    years = dates.year.unique()
    for i in range(len(years) - train_years - test_years + 1):
        train_year = years[i:i+train_years]
        test_year = years[i+train_years:i+train_years+test_years]

        train_idx = dates.year.isin(train_year)
        test_idx = dates.year.isin(test_year)

        yield train_idx, test_idx


@dataclass
class WalkForwardBacktester:
    model: object              # your GNN
    mvo: object
    scaler: object
    pred_lag: int = 1
    name: str = None

    def run(self, X, y, ret):
        holdings = []
        pnl = []

        for train_idx, test_idx in yearly_walkforward_splits(X.index):
            X_train, y_train, ret_train = X.loc[train_idx], y.loc[train_idx], ret.loc[train_idx]
            X_test, y_test, ret_test = X.loc[test_idx], y.loc[test_idx], ret.loc[test_idx]

            # Scale data
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # Train GNN
            self.model.fit(X_train, y_train, ret_train)

            # Predict expected returns (year t+1)
            mu_hat = self.model.predict(X_test)

            self.mvo.fit(X=None, y=ret_train)
            h = self.mvo.predict(mu_hat)

            h = pd.DataFrame(
                h,
                index=ret[test_idx].index,
                columns = ret.columns,
            )

            holdings.append(h)
            pnl.append((h.shift(self.pred_lag) * ret_test).sum(axis=1))

        self.h_ = pd.concat(holdings)
        self.pnl_ = pd.concat(pnl)

        if self.name:
            self.pnl_ = self.pnl_.rename(self.name)

        return self
