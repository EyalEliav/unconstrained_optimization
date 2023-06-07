import unittest
import numpy as np
from tests import examples
from src import unconstrained_min, utils


class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        self.u = unconstrained_min.UnconstrainedOptimizationMinimizer()
        self.minimizers = [
            self.u.newton_direction,
            self.u.gradient_descent,
            self.u.bgfs,
            self.u.sr1
        ]

    def _test_f(self, f, x0=np.array([1, 1], dtype='int64'), max_iter=100):
        for minimizer in self.minimizers:
            print(f"test {f.__name__}  {minimizer.__name__}")
            success, x, f_x = self.u.line_search(minimizer, f, x0, max_iter=max_iter)
        self.assertTrue(success)

    def test_f1(self):
        self._test_f(examples.f1)

    def test_f2(self):
        self._test_f(examples.f2)

    def test_f3(self):
        self._test_f(examples.f3)

    def test_f4(self):
        self._test_f(examples.f4, x0=np.array([-1, 2], dtype='int64'), max_iter=10000)

    def test_f5(self):
        self._test_f(examples.f5)

    def test_f6(self):
        self._test_f(examples.f6)


if __name__ == '__main__':
    unittest.main()
