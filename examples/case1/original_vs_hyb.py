import numpy as np
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def curl_r_ex(pt):
    x, y, z = pt

    first = x*(-24.0*x**3*y**2 + 24.0*x**3*y - 4.0*x**3 - 8.0*x**2*y**3 + 60.0*x**2*y**2 - 52.0*x**2*y + 8.0*x**2 + 12.0*x*y**3 - 42.0*x*y**2 + 30.0*x*y - 4.0*x - 4.0*y**3 + 6.0*y**2 - 2.0*y)
    second = y*(32.0*x**3*y**2 - 48.0*x**3*y + 16.0*x**3 + 6.0*x**2*y**3 - 60.0*x**2*y**2 + 78.0*x**2*y - 24.0*x**2 - 6.0*x*y**3 + 28.0*x*y**2 - 30.0*x*y + 8.0*x + 1.0*y**3 - 2.0*y**2 + 1.0*y)
    third = 0

    return np.vstack((first, second, third))

def div_u_ex(pt):
    x, y, z = pt

    return 2*x*y*(x - 1)*(x*y*(1 - y)*(x - 1) + 4*x*y*(y - 1)**2 - x*(x - 1)*(y - 1)**2 + 4*y*(x - 1)*(y - 1)**2)

def load(N, folder):

    curl_r, div_u, r, u = [], [], [], []

    for n in N:
        curl_r.append(np.loadtxt(folder + "curl_r_" + str(n)))
        div_u.append(np.loadtxt(folder + "div_u_" + str(n)))
        r.append(np.loadtxt(folder + "r_" + str(n)))
        u.append(np.loadtxt(folder + "u_" + str(n)))

    return curl_r, div_u, r, u

def create_grid(n):
    # make the grid
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    network = pp.FractureNetwork2d(domain=domain)

    mesh_size = 1/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

def main():

    keyword="flow"
    N = 2 ** np.arange(4, 9)

    curl_r, div_u, r, u = load(N, "../case1/")
    curl_hat_r, div_hat_u, hat_r, hat_u = load(N, "../case2/")

    h, err_curl_r, err_div_u, err_r, err_u = [], [], [], [], []
    err_ex_curl_r, err_ex_curl_hat_r = [], []
    err_ex_div_u, err_ex_div_hat_u = [], []

    for i, n in enumerate(N):
        mdg = create_grid(n)
        for sd in mdg.subdomains():
            pass

        RT0 = pg.RT0(keyword)
        P0 = pg.PwConstants(keyword)

        face_proj = pg.eval_at_cell_centers(mdg, RT0)
        cell_proj = pg.eval_at_cell_centers(mdg, P0)

        cell_u = (face_proj * u[i]).reshape((3, -1), order="F")
        cell_hat_u = (face_proj * hat_u[i]).reshape((3, -1), order="F")

        cell_curl_r = (face_proj * curl_r[i]).reshape((3, -1), order="F")
        cell_curl_hat_r = (face_proj * curl_hat_r[i]).reshape((3, -1), order="F")

        cell_div_u = cell_proj * div_u[i]
        cell_div_hat_u = cell_proj * div_hat_u[i]

        h.append(error.geometry_info(sd)[0])

        err_r.append(error.ridge(sd, r[i], r_hat = hat_r[i]))
        err_u.append(error.face(sd, cell_u, q_hat = cell_hat_u))

        err_curl_r.append(error.face(sd, cell_curl_r, q_hat = cell_curl_hat_r))
        err_div_u.append(error.cell_center(sd, cell_div_u, p_hat = cell_div_hat_u))

        rt0_curl_r_ex = RT0.interpolate(sd, curl_r_ex)
        err_ex_curl_r.append(error.face_v1(sd, curl_r[i], q_hat = rt0_curl_r_ex))
        err_ex_curl_hat_r.append(error.face_v1(sd, curl_hat_r[i], q_hat = rt0_curl_r_ex))

        err_ex_div_u.append(error.cell(sd, cell_div_u, div_u_ex))
        err_ex_div_hat_u.append(error.cell(sd, cell_div_hat_u, div_u_ex))

    order_r = error.order(err_r, h)
    order_u = error.order(err_u, h)
    order_curl_r = error.order(err_curl_r, h)
    order_div_u = error.order(err_div_u, h)

    order_ex_curl_r = error.order(err_ex_curl_r, h)
    order_ex_curl_hat_r = error.order(err_ex_curl_hat_r, h)
    order_ex_div_u = error.order(err_ex_div_u, h)
    order_ex_div_hat_u = error.order(err_ex_div_hat_u, h)

    print("h\n", h)

    print("err_r\n", err_r)
    print("order_r\n", order_r)

    print("err_u\n", err_u)
    print("order_u\n", order_u)

    print("err_curl_r\n", err_curl_r)
    print("order_curl_r\n", order_curl_r)

    print("err_div_u\n", err_div_u)
    print("order_div_u\n", order_div_u)

    print("err_ex_curl_r\n", err_ex_curl_r)
    print("order_ex_curl_r\n", order_ex_curl_r)

    print("err_ex_curl_hat_r\n", err_ex_curl_hat_r)
    print("order_ex_curl_hat_r\n", order_ex_curl_hat_r)

    print("err_ex_div_u\n", err_ex_div_u)
    print("order_ex_div_u\n", order_ex_div_u)

    print("err_ex_div_hat_u\n", err_ex_div_hat_u)
    print("order_ex_div_hat_u\n", order_ex_div_hat_u)

if __name__ == "__main__":
    main()
