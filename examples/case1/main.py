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

def main(mdg, keyword="flow"):

    # problem data
    mu, labda = 1, 1

    # set the data
    bc_val, bc_ess, v_source = [], [], []
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        bc_ridges = np.zeros(sd.num_ridges)
        bc_faces = np.zeros(sd.num_faces)
        bc_val.append(np.hstack((bc_ridges, bc_faces)))

        ess_ridges = sd.tags["domain_boundary_nodes"]
        ess_faces = sd.tags["domain_boundary_faces"]
        bc_ess.append(np.hstack((ess_ridges, ess_faces)))

        v_source.append(vector_source(sd))

    bc_ess = np.hstack(bc_ess)

    # formulation specific data
    c1 = 2 / mu
    c2 = labda + mu

    # construct the matrices
    ridge_mass = pg.ridge_mass(mdg)
    cell_mass = pg.cell_mass(mdg)

    face_mass = pg.face_mass(mdg)
    curl = face_mass * pg.curl(mdg)
    div = pg.div(mdg)

    div_div = div.T * sps.linalg.inv(cell_mass) * div

    # get the degrees of freedom for each variable
    face_dof, ridge_dof = curl.shape
    dofs = np.cumsum([ridge_dof])

    # assemble the problem
    mat = sps.bmat([[c1*ridge_mass,    -curl.T],
                    [curl,          c2*div_div]], format="csc")

    # assemble the right-hand side
    rhs = np.hstack(bc_val)
    rhs[dofs[0]:] += face_mass * np.hstack(v_source)

    # solve the problem
    ls = pg.LinearSystem(mat, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve()

    # extract the variables
    r, u = np.split(x, dofs)

    # post process rotation
    proj_r = pg.eval_at_cell_centers(mdg, pg.Lagrange(keyword))
    P0r = proj_r * r

    # post process displacement
    proj_u = pg.proj_faces_to_cells(mdg)
    P0u = (proj_u * u).reshape((3, -1), order="F")

    #for _, data in mdg.subdomains(return_data=True):
    #   data[pp.STATE] = {"P0r": P0r, "P0u": P0u}

    #save = pp.Exporter(mdg, "sol")
    #save.write_vtu(["P0r", "P0u"])

    # compute the error
    h, *_ = error.geometry_info(sd)

    err_r = error.ridge(sd, r, r_ex)
    err_q = error.face(sd, P0u, u_ex)

    return h, err_r, err_q

if __name__ == "__main__":

    N = 2 ** np.arange(4, 9)
    err = np.array([main(create_grid(n)) for n in N])

    order_r = error.order(err[:, 1], err[:, 0])
    order_q = error.order(err[:, 2], err[:, 0])

    print("h\n", err[:, 0])

    print("err_r\n", err[:, 1])
    print("order_r\n", order_r)

    print("err_q\n", err[:, 2])
    print("order_q\n", order_q)
