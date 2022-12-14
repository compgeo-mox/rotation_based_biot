import numpy as np
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def curl_r_ex(pt):
    x, y, z = pt

    first = x*(x - 1)*(y**2*z*(y - 1)**2*(z - 1)*(2.0*x*z - 8.0*x*(x - 1) + 2.0*x*(z - 1) + 2.0*z*(x - 1) + 2.0*(x - 1)*(z - 1)) - y**2*z*(y - 1)**2*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) - y**2*(y - 1)**2*(z - 1)*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) - y*z**2*(z - 1)**2*(4.0*x*y*(x - 1) + 2.0*x*y*(y - 1) + 12.0*x*(x - 1)*(y - 1) + 1.0*x*(y - 1)**2 + 2.0*y*(x - 1)*(y - 1) + 1.0*(x - 1)*(y - 1)**2) - z**2*(y - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1)))
    second = y*(y - 1)*(2.0*x**2*z*(x - 1)**2*(z - 1)*(y*z + y*(y - 1) + y*(z - 1) + z*(y - 1) + (y - 1)*(z - 1)) + x**2*z*(x - 1)**2*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x**2*(x - 1)**2*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x*z**2*(z - 1)**2*(8.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 8.0*x*(x - 1)*(y - 1) + 4.0*y*(x - 1)**2 + 3.0*y*(x - 1)*(y - 1) + 4.0*(x - 1)**2*(y - 1)) + z**2*(x - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1)))
    third = z*(z - 1)*(-x**2*y*(x - 1)**2*(y - 1)*(1.0*y*z + 1.0*y*(z - 1) + 1.0*z*(y - 1) + 4.0*z*(z - 1) + 1.0*(y - 1)*(z - 1)) - x**2*y*(x - 1)**2*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) - x**2*(x - 1)**2*(y - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x*y**2*(y - 1)**2*(8.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 8.0*x*(x - 1)*(z - 1) + 4.0*z*(x - 1)**2 - 6.0*z*(x - 1)*(z - 1) + 4.0*(x - 1)**2*(z - 1)) + y**2*(x - 1)*(y - 1)**2*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)))

    return np.vstack((first, second, third))

def div_u_ex(pt):
    x, y, z = pt

    return 2*x*y*z*(x - 1)*(z - 1)*(x*y*z*(1 - y)*(x - 1)*(z - 1) + 2*x*y*z*(x - 1)*(y - 1)**2 + 4*x*y*z*(y - 1)**2*(z - 1) + 2*x*y*(x - 1)*(y - 1)**2*(z - 1) - x*z*(x - 1)*(y - 1)**2*(z - 1) + 4*y*z*(x - 1)*(y - 1)**2*(z - 1))

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
    network = pp.FractureNetwork3d(domain=domain)

    mesh_size = 1/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

def main():

    keyword="flow"
    N = [7, 11, 15, 19, 23]

    curl_r, div_u, r, u = load(N, "../case5/")
    curl_hat_r, div_hat_u, hat_r, hat_u = load(N, "../case6/")

    h, err_curl_r, err_div_u, err_r, err_u = [], [], [], [], []
    err_ex_curl_r, err_ex_curl_hat_r = [], []
    err_ex_div_u, err_ex_div_hat_u = [], []

    for i, n in enumerate(N):
        mdg = create_grid(n)
        for sd in mdg.subdomains():
            pass

        print(r[i].size, u[i].size, r[i].size + u[i].size)

        RT0 = pg.RT0(keyword)
        N0 = pg.Nedelec0(keyword)
        N1 = pg.Nedelec1(keyword)
        P0 = pg.PwConstants(keyword)

        ridge_proj_n0 = pg.eval_at_cell_centers(mdg, N0)
        ridge_proj_n1 = pg.eval_at_cell_centers(mdg, N1)
        face_proj = pg.eval_at_cell_centers(mdg, RT0)
        cell_proj = pg.eval_at_cell_centers(mdg, P0)

        cell_r = (ridge_proj_n0 * r[i]).reshape((3, -1), order="F")
        cell_hat_r = (ridge_proj_n1 * hat_r[i]).reshape((3, -1), order="F")

        cell_u = (face_proj * u[i]).reshape((3, -1), order="F")
        cell_hat_u = (face_proj * hat_u[i]).reshape((3, -1), order="F")

        cell_curl_r = (face_proj * curl_r[i]).reshape((3, -1), order="F")
        cell_curl_hat_r = (face_proj * curl_hat_r[i]).reshape((3, -1), order="F")

        cell_div_u = cell_proj * div_u[i]
        cell_div_hat_u = cell_proj * div_hat_u[i]

        h.append(error.geometry_info(sd)[0])

        err_r.append(error.ridge(sd, cell_r, r_hat = cell_hat_r))
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
