# import libraries
import statsmodels.api as sm  # for beta regression realization
from sklearn.base import BaseEstimator, RegressorMixin  # for adapting a scikitlearn regressor


class BetaRegression(BaseEstimator, RegressorMixin):
    """ Lightweight implementation of a beta regressor using statsmodels library for the implementation
        adapting the scikitlearn regressor base class to be able to use cross validation.
        
        Attributes:
        gamma_model (statsmodels.genmod.generalized_linear_model.GLM): Model of the Beta Regression.
        gamma_params (array): Parameters specifying the gamma model
    """

    def __init__(self):
        """Initialization of the Beta Regression.
        """
        self.gamma_model = None
        self.gamma_params = None

    def fit(self, x, y):
        """Fits the model using maximum likelihood.
        Args:
            x (array-like): A nobs x k array where nobs is the number of observations and k is the number of regressors.
            y (array-like): 1-d endogenous response variable. The dependent variable.
        """
        self.gamma_model = sm.GLM(y, x, family=sm.families.Gamma(link=sm.families.links.log))
        self.gamma_params = self.gamma_model.fit().params
        return self

    def predict(self, x_test):
        """Predicts y for the given exogenous data.
        Args:
            x_test (array-like): A nobs x k array where nobs is the number
            of observations and k is the number of regressors.
        """
        return self.gamma_model.predict(self.gamma_params, x_test)
