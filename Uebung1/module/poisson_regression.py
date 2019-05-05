import statsmodels.api as sm  # for poisson regression
from sklearn.base import BaseEstimator, RegressorMixin

class PoissonRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.poisson_mod_ = None
        self.poisson_res_ = None

    def fit(self, X, y):
        self.poisson_mod_ = sm.Poisson(y, X)
        self.poisson_res_ = self.poisson_mod_.fit(method="newton")
        return self
    def predict(self, X_test):
        return self.poisson_res_.predict(X_test)