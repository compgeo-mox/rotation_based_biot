import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

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
    bc_val, bc_ess, source = [], [], []
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        top_faces = sd.face_centers[1, :] == 1

        faces, _, sign = sps.find(sd.cell_faces)
        sign = sign[np.unique(faces, return_index=True)[1]]

        bc_faces = np.zeros(sd.num_faces)
        bc_faces[top_faces] = -sign[top_faces]
        bc_ridges = np.zeros(sd.num_ridges)
        bc_val.append(np.hstack((bc_ridges, bc_faces, np.zeros(sd.num_cells))))

        ess_faces = np.zeros(sd.num_faces, dtype=bool)
        ess_ridges = np.zeros(sd.num_ridges, dtype=bool)
        ess_cells = np.zeros(sd.num_cells, dtype=bool)
        bc_ess.append(np.hstack((ess_ridges, ess_faces, ess_cells)))

        source.append(np.zeros(sd.num_faces))

    source = np.hstack(source)
    bc_val = np.hstack(bc_val)
    bc_ess = np.hstack(bc_ess)

    # construct the matrices
    M_ridge = pg.ridge_mass(mdg)
    M_cell = pg.cell_mass(mdg)

    face_mass = pg.face_mass(mdg)
    curl = face_mass * pg.curl(mdg)
    div = pg.div(mdg)

    # assemble the saddle point problem
    spp = sps.bmat([[M_ridge, -curl.T, None], [curl, None, -div.T], [None, div, M_cell]], format="csc")

    # get the degrees of freedom for each variable
    _, dof_r = curl.shape
    dof_p, dof_u = div.shape

    # assemble the right-hand side
    rhs = bc_val.copy()
    rhs[dof_r:dof_r+dof_u] += face_mass * source

    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve()

    # extract the variables
    r, u, p = x[:dof_r], x[dof_r:dof_r+dof_u], x[-dof_p:]

    # post process rotation
    proj_r = pg.eval_at_cell_centers(mdg, pg.Lagrange(keyword))
    P0r = proj_r * r

    # post process displacement
    proj_u = pg.proj_faces_to_cells(mdg)
    P0u = (proj_u * u).reshape((3, -1), order="F")

    for sd, data in mdg.subdomains(return_data=True):
       data[pp.STATE] = {"P0r": P0r, "P0u": P0u, "p": p}

    save = pp.Exporter(mdg, "sol")
    save.write_vtu(["P0r", "P0u", "p"])

if __name__ == "__main__":
    main(create_grid(4))
