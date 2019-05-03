import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin


# https://www.ime.usp.br/~sferrari/beta.pdf
# https://www.statsmodels.org/stable/glm.html
# https://gist.github.com/brentp/089c7d6d69d78d26437f

class BetaRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.p = 1.0
        self.q = 1.0

    def fit(self, X, y):
        self.gamma_model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log))
        self.gamma_params = self.gamma_model.fit().params
        return self

    def predict(self, X):
        return self.gamma_model.predict(self.gamma_params, X)