import numpy as np
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def load(N, folder):

    curl_r, div_u, div_q, r, u, q, p = [], [], [], [], [], [], []

    for n in N:
        curl_r.append(np.loadtxt(folder + "curl_r_" + str(n)))
        div_u.append(np.loadtxt(folder + "div_u_" + str(n)))
        div_q.append(np.loadtxt(folder + "div_q_" + str(n)))

        r.append(np.loadtxt(folder + "r_" + str(n)))
        u.append(np.loadtxt(folder + "u_" + str(n)))
        q.append(np.loadtxt(folder + "q_" + str(n)))
        p.append(np.loadtxt(folder + "p_" + str(n)))

    return curl_r, div_u, div_q, r, u, q, p

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
    mu, labda, alpha = 1, 1, 1

    curl_r, div_u, div_q, r, u, q, p = load(N, "../case3/")
    curl_hat_r, div_hat_u, div_hat_q, hat_r, hat_u, hat_q, hat_p = load(N, "../case4/")

    h, err_curl_r, err_div_u, err_div_q, err_r, err_u, err_q, err_p, err_pp = [], [], [], [], [], [], [], [], []
    for i, n in enumerate(N):
        mdg = create_grid(n)
        for sd in mdg.subdomains():
            pass

        print(r[i].size + u[i].size + q[i].size + p[i].size, u[i].size + p[i].size)

        RT0 = pg.RT0(keyword)
        BDM1 = pg.BDM1(keyword)
        P0 = pg.PwConstants(keyword)

        face_proj = pg.eval_at_cell_centers(mdg, RT0)
        face_proj_bdm1 = pg.eval_at_cell_centers(mdg, BDM1)
        cell_proj = pg.eval_at_cell_centers(mdg, P0)

        cell_u = (face_proj * u[i]).reshape((3, -1), order="F")
        cell_hat_u = (face_proj * hat_u[i]).reshape((3, -1), order="F")

        cell_q = (face_proj * q[i]).reshape((3, -1), order="F")
        cell_hat_q = (face_proj_bdm1 * hat_q[i]).reshape((3, -1), order="F")

        cell_p = cell_proj * p[i]
        cell_hat_p = cell_proj * hat_p[i]

        cell_curl_r = (face_proj * curl_r[i]).reshape((3, -1), order="F")
        cell_curl_hat_r = (face_proj * curl_hat_r[i]).reshape((3, -1), order="F")

        cell_div_u = cell_proj * div_u[i]
        cell_div_hat_u = cell_proj * div_hat_u[i]

        cell_div_q = cell_proj * div_q[i]
        cell_div_hat_q = cell_proj * div_hat_q[i]

        cell_pp = (mu + labda) * cell_div_u - alpha * cell_p
        cell_hat_pp = (mu + labda) * cell_div_hat_u - alpha * cell_hat_p

        h.append(error.geometry_info(sd)[0])

        err_r.append(error.ridge(sd, r[i], r_hat = hat_r[i]))
        err_u.append(error.face(sd, cell_u, q_hat = cell_hat_u))
        err_q.append(error.face(sd, cell_q, q_hat = cell_hat_q))
        err_p.append(error.cell_center(sd, cell_p, p_hat = cell_hat_p))
        err_pp.append(error.cell_center(sd, cell_pp, p_hat = cell_hat_pp))

        err_curl_r.append(error.face(sd, cell_curl_r, q_hat = cell_curl_hat_r))
        err_div_u.append(error.cell_center(sd, cell_div_u, p_hat = cell_div_hat_u))
        err_div_q.append(error.cell_center(sd, cell_div_q, p_hat = cell_div_hat_q))

    order_r = error.order(err_r, h)
    order_u = error.order(err_u, h)
    order_q = error.order(err_q, h)
    order_p = error.order(err_p, h)
    order_pp = error.order(err_pp, h)

    order_curl_r = error.order(err_curl_r, h)
    order_div_u = error.order(err_div_u, h)
    order_div_q = error.order(err_div_q, h)

    print("h\n", h)

    print("err_r\n", err_r)
    print("order_r\n", order_r)

    print("err_u\n", err_u)
    print("order_u\n", order_u)

    print("err_q\n", err_q)
    print("order_q\n", order_q)

    print("err_p\n", err_p)
    print("order_p\n", order_p)

    print("err_pp\n", err_pp)
    print("order_pp\n", order_pp)

    print("err_curl_r\n", err_curl_r)
    print("order_curl_r\n", order_curl_r)

    print("err_div_u\n", err_div_u)
    print("order_div_u\n", order_div_u)

    print("err_div_q\n", err_div_q)
    print("order_div_q\n", order_div_q)

if __name__ == "__main__":
    main()
