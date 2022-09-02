import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]
    z = sd.face_centers[2, :]

    first = y*z*(1 - 2*x)*(y - 1)*(z - 1)
    second = 4*x*y**2*z**2*(x - 1)*(z - 1) + 12*x*y**2*z*(x - 1)*(y - 1)**2 + 4*x*y**2*z*(x - 1)*(z - 1)**2 + 12*x*y**2*(x - 1)*(y - 1)**2*(z - 1) + 16*x*y*z**2*(x - 1)*(y - 1)*(z - 1) + 16*x*y*z*(x - 1)*(y - 1)*(z - 1)**2 - x*y*z*(x - 1)*(z - 1) + 4*x*z**2*(x - 1)*(y - 1)**2*(z - 1) + 4*x*z*(x - 1)*(y - 1)**2*(z - 1)**2 - x*z*(x - 1)*(y - 1)*(z - 1) + 4*y**2*z**2*(y - 1)**2*(z - 1) + 4*y**2*z*(y - 1)**2*(z - 1)**2
    third = -4*x*y**2*z**2*(x - 1)*(y - 1) - 16*x*y**2*z*(x - 1)*(y - 1)*(z - 1) - 4*x*y**2*(x - 1)*(y - 1)*(z - 1)**2 - 4*x*y*z**2*(x - 1)*(y - 1)**2 - 12*x*y*z**2*(x - 1)*(z - 1)**2 - 16*x*y*z*(x - 1)*(y - 1)**2*(z - 1) - x*y*z*(x - 1)*(y - 1) - 4*x*y*(x - 1)*(y - 1)**2*(z - 1)**2 - x*y*(x - 1)*(y - 1)*(z - 1) - 12*x*z**2*(x - 1)*(y - 1)*(z - 1)**2 - 4*y**2*z**2*(y - 1)*(z - 1)**2 - 4*y*z**2*(y - 1)**2*(z - 1)**2

    source = np.vstack((first, second, third))
    return np.sum(sd.face_normals * source, axis=0)

def scalar_source(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    return x*y*z*(1 - x)*(1 - y)*(1 - z)

def r_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = 2*x*(x - 1)*(y**2*z**2*(y - 1)**2 + y**2*z**2*(z - 1)**2 + 4*y**2*z*(y - 1)**2*(z - 1) + y**2*(y - 1)**2*(z - 1)**2 + 4*y*z**2*(y - 1)*(z - 1)**2 + z**2*(y - 1)**2*(z - 1)**2)
    second = -2*y*z**2*(y - 1)*(z - 1)**2*(x*y + x*(y - 1) + y*(x - 1) + (x - 1)*(y - 1))
    third = -2*y**2*z*(y - 1)**2*(z - 1)*(x*z + x*(z - 1) + z*(x - 1) + (x - 1)*(z - 1))

    return np.vstack((first, second, third))

def u_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = np.zeros(sd.num_cells)
    second = x*y**2*z**2*(1 - x)*(1 - y)**2*(2*z - 2) + 2*x*y**2*z*(1 - x)*(1 - y)**2*(1 - z)**2
    third = -x*y**2*z**2*(1 - x)*(1 - z)**2*(2*y - 2) - 2*x*y*z**2*(1 - x)*(1 - y)**2*(1 - z)**2

    return np.vstack((first, second, third))

def p_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    z = sd.nodes[2, :]

    return x*y*z*(1 - x)*(1 - y)*(1 - z)

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
    r, u, p = x[:dof_r], x[dof_r:dof_r+dof_u], x[-dof_p:]

    # post process rotation
    proj_r = pg.eval_at_cell_centers(mdg, pg.Nedelec0(keyword))
    P0r = (proj_r * r).reshape((3, -1), order="F")

    # post process displacement
    proj_u = pg.proj_faces_to_cells(mdg)
    P0u = (proj_u * u).reshape((3, -1), order="F")

    #for sd, data in mdg.subdomains(return_data=True):
    #   data[pp.STATE] = {"P0r": P0r, "P0u": P0u, "p": p}

    #save = pp.Exporter(mdg, "sol")
    #save.write_vtu(["P0r", "P0u", "p"])

    # compute the error
    return error.compute(sd, P0r, r_ex, P0u, u_ex, p, p_ex)

if __name__ == "__main__":

    N = np.arange(9, 12) #14
    err = np.array([main(create_grid(n)) for n in N])
    error.order(err)
