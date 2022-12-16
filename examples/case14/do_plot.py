import numpy as np
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

sys.path.append("../../src/")
from mandel import compute_true_solutions, compute_initial_true_solutions, scale_time

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=25)

params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
plt.rcParams.update(params)
mpl.rcParams["axes.linewidth"] = 1.5


def main():
    a, b, c = 100, 10, 1

    # problem data
    mu = 2.475e9
    labda = 1.65e9
    alpha = 1
    c0 = 6.0606e-11
    perm = 9.869e-11
    F = 6e8

    delta = np.sqrt(1e1)
    num_time_steps = 3000 #100*50

    n_alpha = 200
    p_true, u_true = compute_true_solutions(
        [labda, mu, c0, alpha, perm], [F, a, b], n_alpha
    )

    p_0, u_0 = compute_initial_true_solutions(
        [labda, mu, c0, alpha, perm], [F, a, b]
    )

    time = np.arange(num_time_steps) * delta**2
    time_scaled = scale_time(time, [labda, mu, c0, alpha, perm], [F, a, b])

    plot_every = 250
    N = 1000
    x_every = 50 #100

    x = np.linspace(0, a, N)
    y = b/2*np.ones_like(x)
    coord = np.vstack((x, y))

    pvd_file = "sol/sol.pvd"
    root = ET.parse(pvd_file).getroot()[0]
    vtu_files = [r.attrib["file"] for r in root]

    plt.figure(1, figsize=(9, 6))
    colors = plt.gca()
    for t_step in np.arange(1, num_time_steps, plot_every):
        time = t_step * (delta**2)
        p_vals = [p_true(c, time) * a / F * 1e1 for c in coord.T]

        vtu_file = vtu_files[t_step][:-3] + "csv"
        data = np.loadtxt("pol/" + vtu_file, skiprows=1, delimiter=",")

        color = next(colors._get_lines.prop_cycler)['color']
        plt.plot(x / a, p_vals, color=color)

        x_num = data[::x_every, -4] / a
        p_num = data[::x_every, 0]
        plt.plot(x_num[1:-1], p_num[1:-1] * 1e1, marker="o", linestyle="None", color=color)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("$x ~ a^{-1}$")
    plt.ylabel("$p ~ a ~ F^{-1} ~  \\cdot ~ 10$")
    plt.savefig("pressure.pdf", bbox_inches="tight")
    plt.gcf().clear()
    os.system("pdfcrop pressure.pdf pressure.pdf")

if __name__ == "__main__":
    main()
