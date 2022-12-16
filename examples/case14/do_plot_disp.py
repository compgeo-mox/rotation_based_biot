import numpy as np
import sys
import os

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
        u_vals = [u_true(c, time)[0] / a * 1e4 for c in coord.T]

        vtu_file = vtu_files[t_step][:-3] + "csv"
        data = np.loadtxt("pol/" + vtu_file, skiprows=1, delimiter=",")

        color = next(colors._get_lines.prop_cycler)['color']
        plt.plot(x / a, u_vals, color=color)

        x_num = data[::x_every, -4] / a
        u_num = data[::x_every, 6]
        plt.plot(x_num[1:-1], u_num[1:-1] * 1e4, marker="o", linestyle="None", color=color)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("$x ~ a^{-1}$")
    plt.ylabel("$u_x ~ a^{-1} ~ \\cdot ~ 10^{4}$")
    plt.savefig("displacement.pdf", bbox_inches="tight")
    plt.gcf().clear()
    os.system("pdfcrop displacement.pdf displacement.pdf")

    # legend
    file_out = "legend_mandel.pdf"
    plt.figure(2, figsize=(25, 10))

    for t_step in np.arange(1, num_time_steps, plot_every):
        label = "$\\tau = %.2f$" % time_scaled[t_step]
        plt.plot(np.zeros(1), np.zeros(1), label=label)
        plt.xticks([])
    plt.legend(ncol=6, bbox_to_anchor=(1, 0))

    plt.savefig(file_out, bbox_inches="tight")
    plt.gcf().clear()
    plt.clf()
    plt.cla()

    os.system("pdfcrop --margins '0 -560 0 0' " + file_out + " " + file_out)
    os.system("pdfcrop " + file_out + " " + file_out)


if __name__ == "__main__":
    main()
