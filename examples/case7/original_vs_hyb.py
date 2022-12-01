import numpy as np
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

def print_error(name, vals):
    print(name, [eformat(v, 2, 1) for v in vals])

def print_order(name, vals):
    print(name, ["{:0.2f}".format(v) for v in vals])

def curl_r_ex(pt):
    x, y, z = pt

    first = x*(x - 1)*(y**2*z*(y - 1)**2*(z - 1)*(2.0*x*z - 8.0*x*(x - 1) + 2.0*x*(z - 1) + 2.0*z*(x - 1) + 2.0*(x - 1)*(z - 1)) - y**2*z*(y - 1)**2*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) - y**2*(y - 1)**2*(z - 1)*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) - y*z**2*(z - 1)**2*(4.0*x*y*(x - 1) + 2.0*x*y*(y - 1) + 12.0*x*(x - 1)*(y - 1) + 1.0*x*(y - 1)**2 + 2.0*y*(x - 1)*(y - 1) + 1.0*(x - 1)*(y - 1)**2) - z**2*(y - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1)))
    second = y*(y - 1)*(2.0*x**2*z*(x - 1)**2*(z - 1)*(y*z + y*(y - 1) + y*(z - 1) + z*(y - 1) + (y - 1)*(z - 1)) + x**2*z*(x - 1)**2*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x**2*(x - 1)**2*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x*z**2*(z - 1)**2*(8.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 8.0*x*(x - 1)*(y - 1) + 4.0*y*(x - 1)**2 + 3.0*y*(x - 1)*(y - 1) + 4.0*(x - 1)**2*(y - 1)) + z**2*(x - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1)))
    third = z*(z - 1)*(-x**2*y*(x - 1)**2*(y - 1)*(1.0*y*z + 1.0*y*(z - 1) + 1.0*z*(y - 1) + 4.0*z*(z - 1) + 1.0*(y - 1)*(z - 1)) - x**2*y*(x - 1)**2*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) - x**2*(x - 1)**2*(y - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x*y**2*(y - 1)**2*(8.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 8.0*x*(x - 1)*(z - 1) + 4.0*z*(x - 1)**2 - 6.0*z*(x - 1)*(z - 1) + 4.0*(x - 1)**2*(z - 1)) + y**2*(x - 1)*(y - 1)**2*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)))

    return np.vstack((first, second, third))

def div_u_ex(pt):
    x, y, z = pt

    return 2*x*y*z*(x - 1)*(z - 1)*(x*y*z*(1 - y)*(x - 1)*(z - 1) + 2*x*y*z*(x - 1)*(y - 1)**2 + 4*x*y*z*(y - 1)**2*(z - 1) + 2*x*y*(x - 1)*(y - 1)**2*(z - 1) - x*z*(x - 1)*(y - 1)**2*(z - 1) + 4*y*z*(x - 1)*(y - 1)**2*(z - 1))

def div_q_ex(pt):
    x, y, z = pt

    return  -x*y*z*(x - 1)*(z - 1) - x*z*(x - 1)*(y - 1)*(z - 1) - 2*y*np.pi*(y - 1)*np.sin(2*x*np.pi)*np.cos(2*z*np.pi) + 2*np.pi*np.sin(2*y*np.pi)*np.sin(2*z*np.pi)*np.cos(2*x*np.pi)

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
    network = pp.FractureNetwork3d(domain=domain)

    mesh_size = 1/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

def main():

    keyword="flow"
    N = [3, 7, 11, 15, 19, 23]

    curl_r, div_u, div_q, r, u, q, p = load(N, "../case7/")
    curl_hat_r, div_hat_u, div_hat_q, hat_r, hat_u, hat_q, hat_p = load(N, "../case8/")

    h, err_curl_r, err_div_u, err_div_q = [], [], [], []
    err_r, err_u, err_q, err_p = [], [], [], []
    err_ex_curl_r, err_ex_curl_hat_r = [], []
    err_ex_div_u, err_ex_div_hat_u = [], []
    err_ex_div_q, err_ex_div_hat_q = [], []

    for i, n in enumerate(N):
        mdg = create_grid(n)
        for sd in mdg.subdomains():
            pass

        print(r[i].size, u[i].size, r[i].size + u[i].size)

        RT0 = pg.RT0(keyword)
        BDM1 = pg.BDM1(keyword)
        N0 = pg.Nedelec0(keyword)
        N1 = pg.Nedelec1(keyword)
        P0 = pg.PwConstants(keyword)

        ridge_proj_n0 = pg.eval_at_cell_centers(mdg, N0)
        ridge_proj_n1 = pg.eval_at_cell_centers(mdg, N1)
        face_proj_rt0 = pg.eval_at_cell_centers(mdg, RT0)
        face_proj_bdm1 = pg.eval_at_cell_centers(mdg, BDM1)
        cell_proj = pg.eval_at_cell_centers(mdg, P0)

        cell_r = (ridge_proj_n0 * r[i]).reshape((3, -1), order="F")
        cell_hat_r = (ridge_proj_n1 * hat_r[i]).reshape((3, -1), order="F")

        cell_u = (face_proj_rt0 * u[i]).reshape((3, -1), order="F")
        cell_hat_u = (face_proj_rt0 * hat_u[i]).reshape((3, -1), order="F")

        cell_q = (face_proj_rt0 * q[i]).reshape((3, -1), order="F")
        cell_hat_q = (face_proj_bdm1 * hat_q[i]).reshape((3, -1), order="F")

        cell_p = cell_proj * p[i]
        cell_hat_p = cell_proj * hat_p[i]

        cell_curl_r = (face_proj_rt0 * curl_r[i]).reshape((3, -1), order="F")
        cell_curl_hat_r = (face_proj_rt0 * curl_hat_r[i]).reshape((3, -1), order="F")

        cell_div_u = cell_proj * div_u[i]
        cell_div_hat_u = cell_proj * div_hat_u[i]

        cell_div_q = cell_proj * div_q[i]
        cell_div_hat_q = cell_proj * div_hat_q[i]

        h.append(error.geometry_info(sd)[0])

        err_r.append(error.ridge(sd, cell_r, r_hat = cell_hat_r))
        err_u.append(error.face(sd, cell_u, q_hat = cell_hat_u))
        err_q.append(error.face(sd, cell_q, q_hat = cell_hat_q))
        err_p.append(error.cell_center(sd, cell_p, p_hat = cell_hat_p))

        err_curl_r.append(error.face(sd, cell_curl_r, q_hat = cell_curl_hat_r))
        err_div_u.append(error.cell_center(sd, cell_div_u, p_hat = cell_div_hat_u))
        err_div_q.append(error.cell_center(sd, cell_div_q, p_hat = cell_div_hat_q))

        rt0_curl_r_ex = RT0.interpolate(sd, curl_r_ex)
        err_ex_curl_r.append(error.face_v1(sd, curl_r[i], q_hat = rt0_curl_r_ex))
        err_ex_curl_hat_r.append(error.face_v1(sd, curl_hat_r[i], q_hat = rt0_curl_r_ex))

        p0_div_u_ex = P0.interpolate(sd, div_u_ex)
        err_ex_div_u.append(error.cell_center(sd, div_u[i], p_hat = p0_div_u_ex))
        err_ex_div_hat_u.append(error.cell_center(sd, div_hat_u[i], p_hat = p0_div_u_ex))

        p0_div_q_ex = P0.interpolate(sd, div_q_ex)
        err_ex_div_q.append(error.cell_center(sd, div_q[i], p_hat = p0_div_q_ex))
        err_ex_div_hat_q.append(error.cell_center(sd, div_hat_q[i], p_hat = p0_div_q_ex))

    order_r = error.order(err_r, h)
    order_u = error.order(err_u, h)
    order_q = error.order(err_q, h)
    order_p = error.order(err_p, h)

    order_curl_r = error.order(err_curl_r, h)
    order_div_u = error.order(err_div_u, h)
    order_div_q = error.order(err_div_q, h)

    order_ex_curl_r = error.order(err_ex_curl_r, h)
    order_ex_curl_hat_r = error.order(err_ex_curl_hat_r, h)
    order_ex_div_u = error.order(err_ex_div_u, h)
    order_ex_div_hat_u = error.order(err_ex_div_hat_u, h)
    order_ex_div_q = error.order(err_ex_div_q, h)
    order_ex_div_hat_q = error.order(err_ex_div_hat_q, h)

    print("h\n", h)

    print_error("err_r\n", err_r)
    print_order("order_r\n", order_r)

    print_error("err_u\n", err_u)
    print_order("order_u\n", order_u)

    print_error("err_q\n", err_q)
    print_order("order_q\n", order_q)

    print_error("err_p\n", err_p)
    print_order("order_p\n", order_p)

    print_error("err_curl_r\n", err_curl_r)
    print_order("order_curl_r\n", order_curl_r)

    print_error("err_div_u\n", err_div_u)
    print_order("order_div_u\n", order_div_u)

    print_error("err_div_q\n", err_div_q)
    print_order("order_div_q\n", order_div_q)

    print("###############")

    print_error("err_ex_curl_r\n", err_ex_curl_r)
    print_order("order_ex_curl_r\n", order_ex_curl_r)

    print_error("err_ex_curl_hat_r\n", err_ex_curl_hat_r)
    print_order("order_ex_curl_hat_r\n", order_ex_curl_hat_r)

    print_error("err_ex_div_u\n", err_ex_div_u)
    print_order("order_ex_div_u\n", order_ex_div_u)

    print_error("err_ex_div_hat_u\n", err_ex_div_hat_u)
    print_order("order_ex_div_hat_u\n", order_ex_div_hat_u)

    print_error("err_ex_div_q\n", err_ex_div_q)
    print_order("order_ex_div_q\n", order_ex_div_q)

    print_error("err_ex_div_hat_q\n", err_ex_div_hat_q)
    print_order("order_ex_div_hat_q\n", order_ex_div_hat_q)


if __name__ == "__main__":
    main()
