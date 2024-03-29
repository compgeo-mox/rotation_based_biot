import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def elast_vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]

    first = -24.0*x**4*y**2 + 24.0*x**4*y - 4.0*x**4 + 24.0*x**3*y**3 + 12.0*x**3*y**2 - 36.0*x**3*y + 8.0*x**3 - 96.0*x**2*y**4 + 156.0*x**2*y**3 - 66.0*x**2*y**2 + 6.0*x**2*y - 4.0*x**2 + 96.0*x*y**4 - 180.0*x*y**3 + 80.0*x*y**2 + 4.0*x*y - 16.0*y**4 + 32.0*y**3 - 17.0*y**2 + 1.0*y
    second = 24.0*x**4*y**2 - 24.0*x**4*y + 4.0*x**4 - 96.0*x**3*y**3 + 96.0*x**3*y**2 - 8.0*x**3 + 6.0*x**2*y**4 + 132.0*x**2*y**3 - 186.0*x**2*y**2 + 50.0*x**2*y + 3.0*x**2 - 6.0*x*y**4 - 36.0*x*y**3 + 66.0*x*y**2 - 26.0*x*y + 1.0*x + 1.0*y**4 - 2.0*y**3 + 1.0*y**2

    source = np.vstack((first, second, np.zeros(sd.num_faces)))
    return np.sum(sd.face_normals * source, axis=0)

def flow_vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]

    first = x*y*(y - 1) + y*(x - 1)*(y - 1) + np.sin(2*x*np.pi)*np.sin(2*y*np.pi)
    second = x*(x - 1)*(y*(y - 1) + 2*y - 1)

    source = np.vstack((first, second, np.zeros(sd.num_faces)))
    return np.sum(sd.face_normals * source, axis=0)

def flow_scalar_source(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    return 2*x**2*y**2*(1 - y)*(x - 1)**2 + 8*x**2*y**2*(x - 1)*(y - 1)**2 - 2*x**2*y*(x - 1)**2*(y - 1)**2 + 8*x*y**2*(x - 1)**2*(y - 1)**2 + x*y*(x - 1)*(y - 1) + x*y*(x - 1) + x*(x - 1)*(y - 1) + 2*np.pi*np.sin(2*y*np.pi)*np.cos(2*x*np.pi)

def r_ex(sd):
    x, y, z = pt

    return x*y*(1.0*x*y*(1 - x)*(y - 1)**2 + 4.0*x*y*(1 - y)*(x - 1)**2 - 4.0*x*(x - 1)**2*(y - 1)**2 - 1.0*y*(x - 1)**2*(y - 1)**2)

def u_ex(sd):
    x, y, z = pt

    first = 4*x**2*y**2*(1 - x)**2*(1 - y)**2
    second = -x**2*y**2*(1 - x)**2*(1 - y)**2
    return np.vstack((first, second, np.zeros(sd.num_cells)))

def q_ex(sd):
    x, y, z = pt

    first = np.sin(2*x*np.pi)*np.sin(2*y*np.pi)
    second = x*y*(1 - x)*(1 - y)
    return np.vstack((first, second, np.zeros(sd.num_cells)))

def p_ex(sd):
    x, y, z = pt

    return x*y*(1 - x)*(1 - y)

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

def main(n):
    mdg = create_grid(n)

    elast_key = "elasticity"
    flow_key = "flow"

    # problem data
    mu, labda, alpha, c0 = 1, 1, 1, 1
    delta = 1

    # dscretization
    L1 = pg.Lagrange1(elast_key)
    RT0 = pg.RT0(flow_key)
    P0 = pg.PwConstants(flow_key)

    # set the bc data and source terms
    elast_bc_val, elast_bc_ess, elast_v_source = [], [], []
    flow_bc_val, flow_bc_ess, flow_v_source, flow_s_source = [], [], [], []
    for sd, data in mdg.subdomains(return_data=True):
        # boundary identifiers
        bc_faces = np.zeros(sd.num_faces)
        bc_ridges = np.zeros(sd.num_ridges)

        ess_faces = np.zeros(sd.num_faces, dtype=bool)
        ess_ridges = np.zeros(sd.num_ridges, dtype=bool)
        ess_cells = np.zeros(sd.num_cells, dtype=bool)

        # elasticity
        elast_param = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }

        elast_bc_val.append(np.hstack((bc_ridges, bc_faces)))
        elast_bc_ess.append(np.hstack((ess_ridges, ess_faces)))

        elast_v_source.append(elast_vector_source(sd))

        # flow
        flow_param = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }

        flow_bc_val.append(np.hstack((bc_faces, np.zeros(sd.num_cells))))
        flow_bc_ess.append(np.hstack((ess_faces, ess_cells)))

        flow_v_source.append(flow_vector_source(sd))
        flow_s_source.append(flow_scalar_source(sd))

        data[pp.PARAMETERS] = {elast_key: elast_param, flow_key: flow_param}
        data[pp.DISCRETIZATION_MATRICES] = {elast_key: {}, flow_key: {}}

    bc_ess = np.hstack(elast_bc_ess + flow_bc_ess)

    # construct the matrices
    ridge_mass = pg.ridge_mass(mdg)
    cell_mass = pg.cell_mass(mdg)
    face_mass = pg.face_mass(mdg, keyword=elast_key)
    face_perm_mass = pg.face_mass(mdg, keyword=flow_key)

    curl = face_mass * pg.curl(mdg)
    div = cell_mass * pg.div(mdg)
    div_div = pg.div(mdg).T * cell_mass * pg.div(mdg)

    # get the degrees of freedom for each variable
    _, ridge_dof = curl.shape
    cell_dof, face_dof = div.shape
    dofs = np.cumsum([ridge_dof, face_dof, face_dof])

    # assemble the saddle point problem
    spp = sps.bmat([
        [2/mu*ridge_mass,              -curl.T,           None,         None],
        [           curl, (labda + mu)*div_div,           None, -alpha*div.T],
        [           None,                 None, face_perm_mass, -delta*div.T],
        [           None,            alpha*div,      delta*div, c0*cell_mass]
        ], format="csc")

    # assemble the right-hand side
    rhs = np.zeros(spp.shape[0])
    # data from the elastic problem
    rhs[:dofs[1]] += np.hstack(elast_bc_val)
    rhs[dofs[0]:dofs[1]] += face_mass * np.hstack(elast_v_source)
    # data from the flow problem
    rhs[dofs[1]:] += np.hstack(flow_bc_val)
    rhs[dofs[1]:dofs[2]] += face_mass * np.hstack(flow_v_source)
    rhs[dofs[2]:] += np.hstack(flow_s_source)

    # projection matrices
    ridge_proj = pg.eval_at_cell_centers(mdg, L1)
    face_proj = pg.eval_at_cell_centers(mdg, RT0)
    cell_proj = pg.eval_at_cell_centers(mdg, P0)

    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve()

    # extract the variables
    r, u, q, p = np.split(x, dofs)

    # post process rotation
    cell_r = ridge_proj * r

    # post process displacement
    cell_u = (face_proj * u).reshape((3, -1), order="F")

    # post process Darcy velocity
    cell_q = (face_proj * q).reshape((3, -1), order="F")

    # post process Darcy pressure
    cell_p = cell_proj * p

    # compute the error
    h, *_ = error.geometry_info(sd)

    err_r = L1.error_l2(sd, r, r_ex)
    err_u = RT0.error_l2(sd, cell_u, u_ex)
    err_q = error_l2(sd, cell_q, q_ex)
    err_p = error_l2(sd, cell_p, p_ex)

    curl_r = pg.curl(mdg) * r
    div_u = pg.div(mdg) * u
    div_q = pg.div(mdg) * q

    # save some of the info to file
    np.savetxt("curl_r_" + str(n), curl_r)
    np.savetxt("div_u_" + str(n), div_u)
    np.savetxt("div_q_" + str(n), div_q)
    np.savetxt("r_" + str(n), r)
    np.savetxt("u_" + str(n), u)
    np.savetxt("q_" + str(n), q)
    np.savetxt("p_" + str(n), p)

    return h, err_r, err_u, err_q, err_p

if __name__ == "__main__":

    N = 2 ** np.arange(4, 7)

    err = np.array([main(n) for n in N])

    order_r = error.order(err[:, 1], err[:, 0])
    order_u = error.order(err[:, 2], err[:, 0])
    order_q = error.order(err[:, 3], err[:, 0])
    order_p = error.order(err[:, 4], err[:, 0])

    print("h\n", err[:, 0])

    print("err_r\n", err[:, 1])
    print("order_r\n", order_r)

    print("err_u\n", err[:, 2])
    print("order_u\n", order_u)

    print("err_q\n", err[:, 3])
    print("order_q\n", order_u)

    print("err_p\n", err[:, 4])
    print("order_p\n", order_u)
