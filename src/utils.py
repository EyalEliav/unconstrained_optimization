import os

import matplotlib.pyplot as plt
import numpy as np


class GraphDrawing:
    def __init__(self, objective_func, title):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        z = np.zeros(([len(x), len(y)]))

        for i in range(0, len(x)):
            for j in range(0, len(y)):
                z[j, i], _, _ = objective_func(np.array([x[i], y[j]]))

        plt.figure()
        plt.contourf(x, y, z, 20)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.colorbar()

    def draw_point(self, x):
        plt.plot(x[0], x[1], 'go')

    def finish_draw(self, filename):
        plt.show()
        plt.savefig(os.path.dirname(__file__) + f"/../plots/{filename}.png")
