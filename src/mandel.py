from scipy import optimize
import numpy as np


def compute_alpha(factor, n, tol=1e-7):
    """
    Computes the first n positive roots of the function
    f(x) = tan(x) - factor * x
    """

    def f(x):
        return np.tan(x) - factor * x

    alpha = np.zeros(n)

    if factor < 1:
        raise ValueError

    for ind in range(n):
        bracket = (np.array([tol, 1 - tol]) / 2 + ind) * np.pi
        root_result = optimize.root_scalar(f, bracket=bracket, method="bisect")

        if root_result.converged:
            alpha[ind] = root_result.root
        else:
            print("not converged")

    return alpha


def convert_parameters(param):
    labda, mu, c_0, alpha, perm = param

    K = labda + mu * 2.0 / 3.0
    # E = mu * (9 * K) / (3 * K + mu)
    nu = (3 * K - 2 * mu) / (2 * (3 * K + mu))
    K_u = K + alpha**2 / c_0

    B = alpha / (c_0 * K_u)

    nu_u_factor = (alpha * B * (1 - 2 * nu)) / 3
    #nu_u = (nu_u_factor + nu) / (1 - 2 * nu_u_factor)
    nu_u = (3 * nu + B * (1 - 2 * nu)) / (3 - B * (1 - 2 * nu))

    alpha_factor = (1 - nu) / (nu_u - nu)
    #c_f = perm / c_0 * (K + 4.0 / 3.0 * mu) / (K_u + 4.0 / 3.0 * mu)
    c_f = (2 * perm * (B**2) * mu * (1 - nu) * (1 + nu_u) ** 2) / (
        9 * (1 - nu_u) * (nu_u - nu)
    )

    return B, nu, nu_u, c_f, alpha_factor


def true_solution(old_param, bc_param, converted_param, alpha_n):
    mu = old_param[1]
    F, a, _ = bc_param
    B, nu, nu_u, c_f, alpha_factor = converted_param

    def p(x, t):
        result = np.sin(alpha_n) / (alpha_n - np.sin(alpha_n) * np.cos(alpha_n))
        result *= np.cos(alpha_n * x[0] / a) - np.cos(alpha_n)
        result *= np.exp(-np.square(alpha_n) * c_f * t / (a**2))
        return 2 * F * B * (1 + nu_u) / (3 * a) * np.sum(result)

    def u(x, t):
        common_sum = (
            np.sin(alpha_n)
            * np.cos(alpha_n)
            / (alpha_n - np.sin(alpha_n) * np.cos(alpha_n))
        )

        common_sum *= np.exp(-np.square(alpha_n) * c_f * t / (a**2))
        common_sum = np.sum(common_sum)

        x_sum = (
            np.cos(alpha_n)
            / (alpha_n - np.sin(alpha_n) * np.cos(alpha_n))
            * np.sin(alpha_n * x[0] / a)
            * np.exp(-np.square(alpha_n) * c_f * t / (a**2))
        )

        result_x = -F * nu_u / (mu * a) * common_sum
        result_x += F * nu / (2 * mu * a)
        result_x *= x[0]
        result_x += F / mu * np.sum(x_sum)

        result_y = F * (1 - nu_u) / (mu * a) * common_sum
        result_y += -F * (1 - nu) / (2 * mu * a)

        return np.array([result_x, result_y * x[1], 0])

    return p, u

def initial_solution(old_param, bc_param, converted_param):
    mu = old_param[1]
    F, a, _ = bc_param
    B, nu, nu_u, c_f, alpha_factor = converted_param

    def p(x):
        return (F * B * (1 + nu_u)) / (3 * a)

    def u(x):
        ux = ((F * nu_u) / (2 * mu * a)) * x[0]
        uy = ((-F * (1 - nu_u)) / (2 * mu * a)) * x[1]
        return np.array([ux, uy, 0])

    return p, u

def scale_time(t, param, bc_param):
    converted_param = convert_parameters(param)
    c_f = converted_param[-2]
    a = bc_param[1]
    return (t * c_f) / a**2

def compute_true_solutions(param, bc_param, n_alpha=10):
    converted_param = convert_parameters(param)
    alpha_n = compute_alpha(converted_param[-1], n_alpha)
    return true_solution(param, bc_param, converted_param, alpha_n)

def compute_initial_true_solutions(param, bc_param):
    converted_param = convert_parameters(param)
    return initial_solution(param, bc_param, converted_param)

if __name__ == "__main__":

    n_alpha = 10
    # param = [labda, mu, c_0, alpha, perm]
    # p, u = compute_true_solutions(param, bc_param, n_alpha)

    # print(alpha)
