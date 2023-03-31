"""
The :mod:`eval_sympy` module implements the support
to handle SR regression models described as strings.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: BSD-3

import sympy as sym

__all__ = [
    "SymExpr",
]

class SymExpr:
    """Class with support to symbolic manipulation of a
    regression model described as a string"""

    def __init__(self, expr, n_vars):
        '''
        Creates an object containing support methods to
        evaluate and handle symbolic regression models.
        The variables must be labeled "xi" starting from 0.
        E.g.: "sin(x0 + 0.231*x1)"
        Notice that the constant values must have the same 
        precision as the fitted model.

        Parameters
        ----------
        expr : str
                string with the regression model
        n_vars : int
                  number of variables in the training data
        '''
        self.vars = [f"x{i}" for i in range(n_vars)]
        self.expr = sym.sympify(expr)
        self.f = sym.lambdify(self.vars, self.expr, 'numpy')

    def eval(self, X):
        '''
        Evaluates the symbolic regression and returns the
        predictions of some data points X.

        Parameters
        ----------
        X : array-like
              input data points
        '''
        return self.f(*[X[:, i] for i in range(X.shape[1])])
