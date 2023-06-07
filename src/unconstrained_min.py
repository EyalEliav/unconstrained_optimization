import os

import numpy as np

from src.utils import GraphDrawing


class UnconstrainedOptimizationMinimizer:
    def __init__(self, obj_tol=1e-8, param_tol=1e-12):
        self.help_functions = LineSearchCommonFunctions()
        self.obj_tol = obj_tol
        self.param_tol = param_tol

    def check_tol(self, x, x_next, f_x, f_x_next, h_x):
        if np.abs(f_x - f_x_next) < self.obj_tol:
            return True
        if np.linalg.norm(x - x_next) < self.param_tol:
            return True
        return False

    def line_search(self, minimizer, f, x0, max_iter=100):
        """
        generic line search function that receives the minimizing algorithm as parameter.
        it calculates the minimum point with the minimizer received
        """
        iteration = 1
        title = f"{minimizer.__name__} - {f.__name__}"
        graph_drawer = GraphDrawing(f, title)
        with open(os.path.dirname(__file__) + f"/../{minimizer.__name__}", "a") as log_file:
            log_file.write(f"{f.__name__} :\n")
            success = False
            x = x0
            b = np.eye(len(x))
            while not success and iteration < max_iter:
                f_x, _, h_x = f(x)
                x_next, f_x_next, b = minimizer(f, x, b)
                success = self.check_tol(x, x_next, f_x, f_x_next, h_x)
                graph_drawer.draw_point(x)
                line = f"Iteration {iteration}: x={x}, f(x)={f_x}"
                print(line)
                log_file.write(line + "\n")
                x = x_next
                iteration += 1
            graph_drawer.draw_point(x)
            graph_drawer.finish_draw()
            return success, x, f_x

    def gradient_descent(self, f, x, *args):
        """
        minimizing by gradient descent algorithm
        """
        obj_val, gradient_val, _ = f(x)
        direction = -1 * gradient_val
        x_next, f_x_next, g_x_next = self.help_functions.find_next_step(x, f, direction)
        return x_next, f_x_next, 0

    def newton_direction(self, f, x, *args):
        """
        minimizing with the newton direction
        """
        f_x, g_x, h_x = f(x, should_hessian=True)
        if np.count_nonzero(h_x) == 0:
            return x, f_x, 0
        else:
            p = -1 * np.linalg.inv(h_x) @ g_x
        x_next, f_x_next, g_x_next = self.help_functions.find_next_step(x, f, p)
        return x_next, f_x_next, 0

    def bgfs(self, f, x, b):
        """
        minimizing with bgfs
        """
        f_x, g_x, _ = f(x)
        p = -b @ g_x
        x_next, f_x_next, g_x_next = self.help_functions.find_next_step(x, f, p)
        s = x_next - x
        y = g_x_next - g_x
        b += -1 * (b @ s @ s.T * b) / (s.T @ b @ s) + (y @ y.T) / (y.T @ s)
        return x_next, f_x_next, b

    def sr1(self, f, x, b):
        """
        minimizing with sr1
        """
        f_x, g_x, _ = f(x)
        p = -b @ g_x
        x_next, f_x_next, g_x_next = self.help_functions.find_next_step(x, f, p)
        s = x_next - x
        y = g_x_next - g_x
        y_minus_bs = y - b @ s
        b += (y_minus_bs @ y_minus_bs.T) / (y_minus_bs.T @ s)
        return x_next, f_x_next, b


class LineSearchCommonFunctions:
    def find_next_step(self, x, f, direction):
        alpha = 1
        wolfe_conds_set = False
        f_x, gradient_x, _ = f(x)
        while not wolfe_conds_set:
            x_next = x + alpha * direction
            f_x_next, gradient_x_next, _ = f(x_next)
            wolfe_conds_set = self.wolfe_cond(
                f_x, gradient_x, f_x_next, gradient_x_next, direction, alpha
            )
            alpha /= 2
            if alpha == 0:
                print("alpha is zero")
                break
        return x_next, f_x_next, gradient_x_next

    def wolfe_cond(self, f_x, g_x, f_x_next, g_x_next, direction, alpha):
        return f_x_next <= f_x + 0.5 * alpha * g_x.T @ direction
