import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def elast_vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]
    z = sd.face_centers[2, :]

    first = -16*x**2*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1) + 8*x**2*y**2*z**2*(x - 1)*(y - 1)*(z - 1)**2 - 16*x**2*y**2*z**2*(y - 1)**2*(z - 1)**2 - 16*x**2*y**2*z*(x - 1)*(y - 1)**2*(z - 1)**2 + 8*x**2*y*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 - 16*x*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1)**2 - 64*x*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 - 16*x*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 + x*y**2*z*(x - 1)*(y - 1)**2*(z - 1)*(2.0*x*z - 8.0*x*(x - 1) + 2.0*x*(z - 1) + 2.0*z*(x - 1) + 2.0*(x - 1)*(z - 1)) - x*y**2*z*(x - 1)*(y - 1)**2*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) - x*y**2*(x - 1)*(y - 1)**2*(z - 1)*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) + 8*x*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - x*y*z**2*(x - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 2.0*x*y*(y - 1) + 12.0*x*(x - 1)*(y - 1) + 1.0*x*(y - 1)**2 + 2.0*y*(x - 1)*(y - 1) + 1.0*(x - 1)*(y - 1)**2) - x*y*z*(y - 1)*(z - 1) - x*z**2*(x - 1)*(y - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1)) - 16*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - y*z*(x - 1)*(y - 1)*(z - 1)
    second = -16*x**2*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1) + 4*x**2*y**2*z**2*(x - 1)**2*(z - 1)**2 - 32*x**2*y**2*z**2*(x - 1)*(y - 1)*(z - 1)**2 - 16*x**2*y**2*z*(x - 1)**2*(y - 1)*(z - 1)**2 - 16*x**2*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 16*x**2*y*z**2*(x - 1)**2*(y - 1)*(z - 1)**2 - 32*x**2*y*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 - 16*x**2*y*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 + 2.0*x**2*y*z*(x - 1)**2*(y - 1)*(z - 1)*(y*z + y*(y - 1) + y*(z - 1) + z*(y - 1) + (y - 1)*(z - 1)) + x**2*y*z*(x - 1)**2*(y - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x**2*y*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + 4*x**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - 32*x*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1)**2 - 32*x*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 + x*y*z**2*(y - 1)*(z - 1)**2*(8.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 8.0*x*(x - 1)*(y - 1) + 4.0*y*(x - 1)**2 + 3.0*y*(x - 1)*(y - 1) + 4.0*(x - 1)**2*(y - 1)) - x*y*z*(x - 1)*(z - 1) - x*z*(x - 1)*(y - 1)*(z - 1) + y*z**2*(x - 1)*(y - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1))
    third = -8*x**2*y**2*z**2*(x - 1)**2*(y - 1)**2 + 8*x**2*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1) - 32*x**2*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1) - 32*x**2*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x**2*y**2*z*(x - 1)**2*(y - 1)*(z - 1)**2 - 32*x**2*y**2*z*(x - 1)*(y - 1)**2*(z - 1)**2 - 8*x**2*y**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 + 8*x**2*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x**2*y*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 - x**2*y*z*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z + 1.0*y*(z - 1) + 1.0*z*(y - 1) + 4.0*z*(z - 1) + 1.0*(y - 1)*(z - 1)) - x**2*y*z*(x - 1)**2*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) - x**2*z*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) - 32*x*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1) - 32*x*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 + x*y**2*z*(y - 1)**2*(z - 1)*(8.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 8.0*x*(x - 1)*(z - 1) + 4.0*z*(x - 1)**2 - 6.0*z*(x - 1)*(z - 1) + 4.0*(x - 1)**2*(z - 1)) - x*y*z*(x - 1)*(y - 1) - x*y*(x - 1)*(y - 1)*(z - 1) + y**2*z*(x - 1)*(y - 1)**2*(z - 1)*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1))

    source = np.vstack((first, second, third))
    return np.sum(sd.face_normals * source, axis=0)

def flow_vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]
    z = sd.face_centers[2, :]

    first = -x*y*z*(y - 1)*(z - 1) - y*z*(x - 1)*(y - 1)*(z - 1) + np.sin(2*x*np.pi)*np.sin(2*y*np.pi)*np.sin(2*z*np.pi)
    second = x*z*(x - 1)*(z - 1)*(-y*(y - 1) - 2*y + 1)
    third = -y*(y - 1)*(x*z*(x - 1) + x*(x - 1)*(z - 1) + np.sin(2*x*np.pi)*np.sin(2*z*np.pi))

    source = np.vstack((first, second, third))
    return np.sum(sd.face_normals * source, axis=0)

def flow_scalar_source(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    return 2*x**2*y**2*z**2*(1 - y)*(x - 1)**2*(z - 1)**2 + 4*x**2*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x**2*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 + 4*x**2*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 - 2*x**2*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 + 8*x*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - x*y*z*(x - 1)*(y - 1)*(z - 1) - x*y*z*(x - 1)*(z - 1) - x*z*(x - 1)*(y - 1)*(z - 1) - 2*y*np.pi*(y - 1)*np.sin(2*x*np.pi)*np.cos(2*z*np.pi) + 2*np.pi*np.sin(2*y*np.pi)*np.sin(2*z*np.pi)*np.cos(2*x*np.pi)

def r_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = x**2*y*z*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1))
    second = x*y**2*z*(y - 1)**2*(z - 1)*(2.0*x*z*(1 - x)*(z - 1) + 4.0*x*z*(x - 1)**2 + 4.0*x*(x - 1)**2*(z - 1) - 2.0*z*(x - 1)**2*(z - 1))
    third = x*y*z**2*(z - 1)**2*(1.0*x*y*(1 - x)*(y - 1)**2 + 4.0*x*y*(1 - y)*(x - 1)**2 - 4.0*x*(x - 1)**2*(y - 1)**2 - 1.0*y*(x - 1)**2*(y - 1)**2)

    return np.vstack((first, second, third))

def u_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = 4*x**2*y**2*z**2*(1 - x)**2*(1 - y)**2*(1 - z)**2
    second = -x**2*y**2*z**2*(1 - x)**2*(1 - y)**2*(1 - z)**2
    third = 2*x**2*y**2*z**2*(1 - x)**2*(1 - y)**2*(1 - z)**2

    return np.vstack((first, second, third))

def q_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = np.sin(2*x*np.pi)*np.sin(2*y*np.pi)*np.sin(2*z*np.pi)
    second = x*y*z*(1 - x)*(1 - y)*(1 - z)
    third = y*(1 - y)*np.sin(2*x*np.pi)*np.sin(2*z*np.pi)

    return np.vstack((first, second, third))

def p_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    z = sd.nodes[2, :]

    return x*y*(1 - x)*(1 - y)*z*(1-z)

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

def main(n):
    mdg = create_grid(n)

    elast_key = "elasticity"
    flow_key = "flow"

    # problem data
    mu, labda, alpha, c0 = 1, 1, 1, 1
    delta = 1

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

    # dscretization
    N0 = pg.Nedelec0(elast_key)
    RT0 = pg.RT0(flow_key)
    P0 = pg.PwConstants(flow_key)

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

    print(ridge_dof, face_dof, face_dof, cell_dof, ridge_dof + face_dof + face_dof + cell_dof)

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
    ridge_proj = pg.eval_at_cell_centers(mdg, N0)
    face_proj = pg.eval_at_cell_centers(mdg, RT0)
    cell_proj = pg.eval_at_cell_centers(mdg, P0)

    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve()

    # extract the variables
    r, u, q, p = np.split(x, dofs)

    # post process rotation
    cell_r = (ridge_proj * r).reshape((3, -1), order="F")

    # post process displacement
    cell_u = (face_proj * u).reshape((3, -1), order="F")

    # post process Darcy velocity
    cell_q = (face_proj * q).reshape((3, -1), order="F")

    # post process Darcy pressure
    cell_p = cell_proj * p

    # compute the error
    h, *_ = error.geometry_info(sd)

    err_r = error.ridge(sd, cell_r, r_ex)
    err_u = error.face(sd, cell_u, u_ex)
    err_q = error.face(sd, cell_q, q_ex)
    err_p = error.cell(sd, cell_p, p_ex)

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

    N = np.arange(9, 14)
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
    print("order_q\n", order_q)

    print("err_p\n", err[:, 4])
    print("order_p\n", order_p)
