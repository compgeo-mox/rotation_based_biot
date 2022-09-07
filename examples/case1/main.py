import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]

    first = 24*x**4*y - 12*x**4 - 48*x**3*y + 24*x**3 + 48*x**2*y**3 - 72*x**2*y**2 + 48*x**2*y - 12*x**2 - 48*x*y**3 + 74*x*y**2 - 26*x*y + 8*y**3 - 13*y**2 + 5*y

    second = -48*x**3*y**2 + 48*x**3*y - 8*x**3 + 72*x**2*y**2 - 70*x**2*y + 11*x**2 - 24*x*y**4 + 48*x*y**3 - 48*x*y**2 + 22*x*y - 3*x + 12*y**4 - 24*y**3 + 12*y**2

    source = np.vstack((first, second, np.zeros(sd.num_faces)))
    return np.sum(sd.face_normals * source, axis=0)

def scalar_source(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    return  x*y*(1 - x)*(1 - y)

def r_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    return 12*x**2*y**2*(x - 1)**2 + 12*x**2*y**2*(y - 1)**2 - 12*x**2*y*(x - 1)**2 + 2*x**2*(x - 1)**2 - 12*x*y**2*(y - 1)**2 + 2*y**2*(y - 1)**2

def u_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    first = -2 * x * y * (x - 1) * (y - 1) * x * (x - 1) * (2 * y - 1)
    second = 2 * x * y * (x - 1) * (y - 1) * y * (y - 1) * (2 * x - 1)
    return np.vstack((first, second, np.zeros(sd.num_cells)))

def p_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    return  x*y*(1 - x)*(1 - y)

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
    # set the data
    bc_val, bc_ess, v_source, s_source = [], [], [], []
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        bc_faces = np.zeros(sd.num_faces)
        bc_ridges = np.zeros(sd.num_ridges)
        bc_val.append(np.hstack((bc_ridges, bc_faces, np.zeros(sd.num_cells))))

        ess_faces = np.zeros(sd.num_faces, dtype=bool)
        ess_ridges = np.zeros(sd.num_ridges, dtype=bool)
        ess_cells = np.zeros(sd.num_cells, dtype=bool)
        bc_ess.append(np.hstack((ess_ridges, ess_faces, ess_cells)))

        v_source.append(vector_source(sd))
        s_source.append(scalar_source(sd))

    v_source = np.hstack(v_source)
    s_source = np.hstack(s_source)
    bc_val = np.hstack(bc_val)
    bc_ess = np.hstack(bc_ess)

    # construct the matrices
    ridge_mass = pg.ridge_mass(mdg)
    cell_mass = pg.cell_mass(mdg)

    face_mass = pg.face_mass(mdg)
    curl = face_mass * pg.curl(mdg)
    div = pg.div(mdg)

    # assemble the saddle point problem
    spp = sps.bmat([[ridge_mass, -curl.T,      None],
                    [curl,          None,    -div.T],
                    [None,         -div,  cell_mass]], format="csc")

    # get the degrees of freedom for each variable
    _, dof_r = curl.shape
    dof_p, dof_u = div.shape

    # assemble the right-hand side
    rhs = bc_val.copy()
    rhs[dof_r:dof_r+dof_u] += face_mass * v_source
    rhs[-dof_p:] += cell_mass * s_source

    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve()

    # extract the variables
    r, u, p = np.split(x, np.cumsum([dof_r, dof_u]))

    # post process rotation
    proj_r = pg.eval_at_cell_centers(mdg, pg.Lagrange(keyword))
    P0r = proj_r * r

    # post process displacement
    proj_u = pg.proj_faces_to_cells(mdg)
    P0u = (proj_u * u).reshape((3, -1), order="F")

    #for sd, data in mdg.subdomains(return_data=True):
    #   data[pp.STATE] = {"P0r": P0r, "P0u": P0u, "p": p}

    #save = pp.Exporter(mdg, "sol")
    #save.write_vtu(["P0r", "P0u", "p"])

    # compute the error
    return error.compute(sd, r, r_ex, P0u, u_ex, p, p_ex)

if __name__ == "__main__":

    N = 2 ** np.arange(4, 9)
    err = np.array([main(create_grid(n)) for n in N])
    error.order(err)
