# import libraries
import statsmodels.api as sm  # for poisson regression realization
from sklearn.base import BaseEstimator, RegressorMixin  # for adapting a scikitlearn regressor


class PoissonRegression(BaseEstimator, RegressorMixin):
    """ Lightweight implementation of a poisson regressor using statsmodels library for the implementation
        adapting the scikitlearn regressor base class to be able to use cross validation.
        
        Attributes:
        poisson_mod (statsmodels.discrete.discrete_model.Poisson): Model of the Poisson Regression.
        poisson_res (statsmodels.regression.linear_model.RegressionResults): Result of the Poisson Regression.
        method (str, optional): The method determines which solver from scipy.optimize is used.
    """

    def __init__(self, method="newton"):
        """Initialization of the Poisson Regression.

        Args:
            method (str): The method determines which solver from scipy.optimize is used,
                and it can be chosen from among the following strings:
                "newton" for Newton-Raphson, "nm" for Nelder-Mead
                "bfgs" for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                "lbfgs" for limited-memory BFGS with optional box constraints
                "powell" for modified Powell"s method
                "cg" for conjugate gradient
                "ncg" for Newton-conjugate gradient
                "basinhopping" for global basin-hopping solver
                "minimize" for generic wrapper of scipy minimize (BFGS by default)
        """
        self.poisson_mod = None  # init instance attributes in constructor
        self.poisson_res = None
        self.method = method

    def fit(self, X, y):
        """Fits the model using maximum likelihood.
        Args:
            X (array-like): A nobs x k array where nobs is the number of observations and k is the number of regressors.
            y (array-like): 1-d endogenous response variable. The dependent variable.
        """
        self.poisson_mod = sm.Poisson(y, X)
        self.poisson_res = self.poisson_mod.fit(method=self.method)
        return self

    def predict(self, X_test):
        """Predicts y for the given exogenous data.
        Args:
            X_test (array-like): A nobs x k array where nobs is the number of observations and k is the number of regressors.
        """
        return self.poisson_res.predict(X_test)
