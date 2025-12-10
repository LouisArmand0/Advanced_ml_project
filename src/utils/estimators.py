from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, Ridge
def add_transform_method(cls):
    def transform(self, X):
        return self.predict(X)
    cls.transform = transform
    return cls

@add_transform_method
class LinearRegression(LinearRegression):
    pass
