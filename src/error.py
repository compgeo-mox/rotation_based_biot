import numpy as np
import scipy.sparse as sps

import pygeon as pg

def geometry_info(sd):
    return np.mean(sd.cell_diameters()), sd.num_cells, sd.num_faces, sd.num_ridges

def ridge(sd, r, r_ex):
    if sd.dim == 3:
        norm_r = np.sqrt(
            np.trace((r_ex(sd) @ sps.diags(sd.cell_volumes) @ r_ex(sd).T))
        )
        err_r = np.sqrt(
            np.trace(((r_ex(sd) - r) @ sps.diags(sd.cell_volumes) @ (r_ex(sd) - r).T))
        ) / norm_r
    else:
        mass = pg.Lagrange1("flow").assemble_mass_matrix(sd, None)
        norm_r = np.sqrt(r_ex(sd) @ mass @ r_ex(sd).T)
        err_r = np.sqrt((r_ex(sd) - r) @ mass @ (r_ex(sd) - r).T) / norm_r
    return err_r


def face(sd, P0q, q_ex):
    norm_q = np.sqrt(
        np.trace(q_ex(sd) @ sps.diags(sd.cell_volumes) @ q_ex(sd).T)
    )
    err_q = np.sqrt(
        np.trace((q_ex(sd) - P0q) @ sps.diags(sd.cell_volumes) @ (q_ex(sd) - P0q).T)
    ) / norm_q
    return err_q

def cell(sd, p, p_ex):
    #norm_p = np.sqrt(p_ex(sd) @ sps.diags(sd.cell_volumes) @ p_ex(sd).T)
    #err_p = np.sqrt((p_ex(sd) - p) @ sps.diags(sd.cell_volumes) @ (p_ex(sd) - p).T) / norm_p
    norm_p = _cell_error(sd, np.zeros(sd.num_cells), p_ex(sd))
    err_p = _cell_error(sd, p, p_ex(sd)) / norm_p
    return err_p

def order(err, h):
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])

def _cell_error(sd, sol, sol_ex):
    cell_nodes = sd.cell_nodes()
    err = 0
    for c in np.arange(sd.num_cells):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
        nodes_loc = cell_nodes.indices[loc]
        err_loc = sol_ex[nodes_loc] - sol[c]

        err += sd.cell_volumes[c] * err_loc @ err_loc.T / (sd.dim + 1)
    return np.sqrt(err)
