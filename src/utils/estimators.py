from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from lightgbm.sklearn import LGBMRegressor


def add_transform_method(cls):
    def transform(self, X):
        return self.predict(X)
    cls.transform = transform
    return cls

@add_transform_method
class LinearRegression(LinearRegression):
    pass

@add_transform_method
class MultiOutputRegressor(MultiOutputRegressor):
    pass

class MultiLGBMRegressor(MultiOutputRegressor):
    """
    Multi-output extension of the lightgbm regressor as a transform class
    get_params and set_params attributes necessary for cloning the class
    """

    def __init__(self, **kwargs):
        if "n_jobs" in kwargs.keys():
            kwargs["n_jobs"] = 1

        else:
            kwargs = {"n_jobs": 1, **kwargs}
        self.m = MultiOutputRegressor(LGBMRegressor(**kwargs))

    def get_params(self, deep=True):
        return self.m.estimator.get_params(deep=deep)

    def set_params(self, **kwargs):
        if "n_jobs" in kwargs.keys():
            kwargs["n_jobs"] = 1

        else:
            kwargs = {"n_jobs": 1, **kwargs}
        return self.m.estimator.set_params(**kwargs)

    def fit(self, X, y):
        return self.m.fit(X, y)

    def transform(self, X):
        return self.m.transform(X)