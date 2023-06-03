import numpy as np


class UnconstrainedOptimizationMinimizer:

    def line_search(self, minimizer, f, x0, max_iter=100):
        """
        generic line search function that receives the minimizing algorithm as parameter.
        it calculates the minimum point with the minimizer received
        """

    def gradient_descent(self, f, x, *args):
        """
        minimizing by gradient descent algorithm
        """

    def newton_direction(self, f, x, *args):
        """
        minimizing with the newton direction
        """

    def bgfs(self, f, x, b):
        """
        minimizing with bgfs
        """

    def sr1(self, f, x, b):
        """
        minimizing with sr1
        """