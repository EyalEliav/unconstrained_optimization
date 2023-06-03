import numpy as np


class UnconstrainedOptimizationMinimizer:

    def __init__(self):
        self.help_functions = LineSearchCommonFunctions()

    def line_search(self, minimizer, f, x0, max_iter=100):
        """
        generic line search function that receives the minimizing algorithm as parameter.
        it calculates the minimum point with the minimizer received
        """

    def find_next_step(self):

    def gradient_descent(self, f, x, *args):
        """
        minimizing by gradient descent algorithm
        """
        obj_val, gradient_val, hessien_val = f(x)
        direction = -1 * gradient_val
        x_next, f_x_next, g_x_next = self.help_functions.find_next_step(x, f, direction)
        return x_next, f_x_next, 0

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


class LineSearchCommonFunctions:

    def find_next_step(self, x, f, direction):
        alpha = 1
        wolfe_conds_set = False
        f_x, gradient_x, _ = f(x)
        while not wolfe_conds_set:
            x_next = x + alpha * direction
            f_x_next, gradient_x_next, _ = f(x_next)
            wolfe_conds_set = self.wolfe_conds(f_x, gradient_x, f_x_next, gradient_x_next, direction, alpha)
            alpha /= 2
            if alpha == 0:
                print("alpha is zero")
                break
        return x_next, f_x_next, gradient_x_next

    def wolfe_conds(self, f_x, g_x, f_x_next, g_x_next, direction, alpha):
        cond1 = f_x_next <= f_x + 0.5 * alpha * g_x.T @ direction
        cond2 = g_x_next.T @ p >= 0.5 * g_x.T @ direction
        return cond1 and cond2