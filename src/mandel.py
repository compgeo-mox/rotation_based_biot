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


def convert_parameters(labda, mu, c_0, alpha, perm, n_alpha):

    K = labda + mu * 2.0 / 3.0
    # E = mu * (9 * K) / (3 * K + mu)
    nu = (3 * K - 2 * mu) / (2 * (3 * K + mu))
    K_u = K + alpha**2 / c_0

    B = alpha / (c_0 * K_u)

    nu_u_factor = (alpha * B * (1 - 2 * nu)) / 3
    nu_u = (nu_u_factor + nu) / (1 - 2 * nu_u_factor)

    alpha_factor = (1 - nu) / (nu_u - nu)
    c_f = perm / c_0 * (K + 4.0 / 3.0 * mu) / (K_u + 4.0 / 3.0 * mu)
    alpha_n = compute_alpha(alpha_factor, n_alpha)

    return B, nu, nu_u, c_f, alpha_n


def true_solution(old_param, bc_param, converted_param):
    labda, mu, c_0, alpha, perm, n_alpha = old_param
    F, a, b = bc_param
    B, nu, nu_u, c_f, alpha_n = converted_param

    def p(x, t):
        result = np.sin(alpha_n) / (alpha_n - np.sin(alpha_n) * np.cos(alpha_n))
        result *= np.cos(alpha_n * x / a) - np.cos(alpha_n)
        result *= np.exp(-(alpha_n**2) * c_f * t / (a**2))
        return 2 * F * B * (1 + nu_u) / (3 * a) * np.sum(result)

    def u(x, t):
        result_x = (
            np.sin(alpha_n)
            * np.cos(alpha_n)
            / (alpha_n - np.sin(alpha_n) * np.cos(alpha_n))
        )
        result_x *= np.exp(-(alpha_n**2) * c_f * t / (a**2))
        result_x = -F * nu_u / (mu * a) * sum(result_x)
        result_x += F * nu / (2 * mu * a)

        # TODO
        result_y = 0

        return result_x * x[0], result_y * x[1]

    return p, u


if __name__ == "__main__":
    alpha = compute_alpha(2.0, 5)
    print(alpha)
