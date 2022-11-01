import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]

    first = -24.0*x**4*y**2 + 24.0*x**4*y - 4.0*x**4 + 24.0*x**3*y**3 + 12.0*x**3*y**2 - 36.0*x**3*y + 8.0*x**3 - 96.0*x**2*y**4 + 156.0*x**2*y**3 - 66.0*x**2*y**2 + 6.0*x**2*y - 4.0*x**2 + 96.0*x*y**4 - 180.0*x*y**3 + 78.0*x*y**2 + 6.0*x*y - 16.0*y**4 + 32.0*y**3 - 16.0*y**2
    second = 24.0*x**4*y**2 - 24.0*x**4*y + 4.0*x**4 - 96.0*x**3*y**3 + 96.0*x**3*y**2 - 8.0*x**3 + 6.0*x**2*y**4 + 132.0*x**2*y**3 - 186.0*x**2*y**2 + 48.0*x**2*y + 4.0*x**2 - 6.0*x*y**4 - 36.0*x*y**3 + 66.0*x*y**2 - 24.0*x*y + 1.0*y**4 - 2.0*y**3 + 1.0*y**2

    source = np.vstack((first, second, np.zeros(sd.num_faces)))
    return np.sum(sd.face_normals * source, axis=0)

def r_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    return x*y*(1.0*x*y*(1 - x)*(y - 1)**2 + 4.0*x*y*(1 - y)*(x - 1)**2 - 4.0*x*(x - 1)**2*(y - 1)**2 - 1.0*y*(x - 1)**2*(y - 1)**2)

def u_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]

    first = 4*x**2*y**2*(1 - x)**2*(1 - y)**2
    second = -x**2*y**2*(1 - x)**2*(1 - y)**2
    return np.vstack((first, second, np.zeros(sd.num_cells)))

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

def main(n, keyword="flow"):
    mdg = create_grid(n)

    # problem data
    mu, labda = 1, 1

    # set the data
    bc_val, ess_faces, ess_ridges, v_source = [], [], [], []
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        bc_val.append(np.zeros(sd.num_faces))

        ess_faces.append(sd.tags["domain_boundary_faces"])
        ess_ridges.append(sd.tags["domain_boundary_nodes"])

        v_source.append(vector_source(sd))

    ess_faces = np.hstack(ess_faces)
    ess_ridges = np.hstack(ess_ridges)

    # dscretization
    L1 = pg.Lagrange1(keyword)
    RT0 = pg.RT0(keyword)

    # construct the matrices
    lumped_ridge_mass = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, None)
    cell_mass = pg.cell_mass(mdg)

    face_mass = pg.face_mass(mdg)

    curl = face_mass * pg.curl(mdg)
    div = cell_mass * pg.div(mdg)
    div_div = pg.div(mdg).T * cell_mass * pg.div(mdg)

    # assemble the problem
    ls = pg.LinearSystem(2/mu*lumped_ridge_mass, curl.T.tocsc())
    ls.flag_ess_bc(ess_ridges, np.zeros(ess_ridges.size))
    A = ls.solve()

    mat = curl * A + (labda + mu)*div_div

    # assemble the right-hand side
    rhs = face_mass * np.hstack(v_source)

    # solve the problem
    ls = pg.LinearSystem(mat, rhs)
    ls.flag_ess_bc(ess_faces, np.zeros(ess_faces.size))
    u = ls.solve()

    # projection matices
    ridge_proj = pg.eval_at_cell_centers(mdg, L1)
    face_proj = pg.eval_at_cell_centers(mdg, RT0)

    # post process rotation
    r = A * u
    cell_r = ridge_proj * r

    # post process displacement
    cell_u = (face_proj * u).reshape((3, -1), order="F")

    # compute the error
    h, *_ = error.geometry_info(sd)

    err_r = error.ridge(sd, r, r_ex)
    err_u = error.face(sd, cell_u, u_ex)

    curl_r = pg.curl(mdg) * r
    div_u = pg.div(mdg) * u

    # save some of the info to file
    np.savetxt("curl_r_" + str(n), curl_r)
    np.savetxt("div_u_" + str(n), div_u)
    np.savetxt("r_" + str(n), r)
    np.savetxt("u_" + str(n), u)

    return h, err_r, err_u

if __name__ == "__main__":

    N = 2 ** np.arange(4, 9)
    err = np.array([main(n) for n in N])

    order_r = error.order(err[:, 1], err[:, 0])
    order_u = error.order(err[:, 2], err[:, 0])

    print("h\n", err[:, 0])

    print("err_r\n", err[:, 1])
    print("order_r\n", order_r)

    print("err_u\n", err[:, 2])
    print("order_u\n", order_u)
