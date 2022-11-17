from scipy import optimize
import numpy as np


def compute_alpha(factor, n):
    """
    Computes the first n positive roots of the function
    f(x) = tan(x) - factor * x
    """

    def f(x):
        return np.tan(x) - factor * x

    alpha = np.zeros(n)

    for ind in range(n):
        bracket = (np.array([0, 1 - 1e-7]) / 2 + ind + 1) * np.pi
        root_result = optimize.root_scalar(f, bracket=bracket, method="bisect")

        if root_result.converged:
            alpha[ind] = root_result.root

    return alpha


if __name__ == "__main__":
    alpha = compute_alpha(2.0, 5)
    print(alpha)
