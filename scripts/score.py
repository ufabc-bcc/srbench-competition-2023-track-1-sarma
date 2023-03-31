"""
The :mod:`calculate_metric` module implements the calculation
of the evaluation metrics.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#
# License: BSD-3

from scripts.symexpr import SymExpr
from sklearn.metrics import r2_score

__all__ = [
    "Score",
]

class Score:
    """Class with support to symbolic manipulation of a
    regression model described as a string"""

    def __init__(self, expr, X, y):
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
        self.expr = SymExpr(expr, X.shape[1])
        self.r2 = r2_score(y, self.expr.eval(X))
        self.n_nodes = self.count_nodes(self.expr.expr)

    def count_nodes(self, expr):
        '''
        Counts the nodes of a sympy expression.

        Parameters
        ----------
        expr : sympy
                 sympy expression as created by SymExpr class.
        '''
        count = 0
        for arg in expr.args:
            count += self.count_nodes(arg)
        return count + 1
