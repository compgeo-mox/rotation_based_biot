import numpy as np
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

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

    order_r = error.order(err_r, h)
    order_u = error.order(err_u, h)
    order_curl_r = error.order(err_curl_r, h)
    order_div_u = error.order(err_div_u, h)

    print("h\n", h)

    print("err_r\n", err_r)
    print("order_r\n", order_r)

    print("err_u\n", err_u)
    print("order_u\n", order_u)

    print("err_curl_r\n", err_curl_r)
    print("order_curl_r\n", order_curl_r)

    print("err_div_u\n", err_div_u)
    print("order_div_u\n", order_div_u)

if __name__ == "__main__":
    main()
