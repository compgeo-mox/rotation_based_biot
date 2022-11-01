import numpy as np
import scipy.sparse as sps

import pygeon as pg

def geometry_info(sd):
    return np.mean(sd.cell_diameters()), sd.num_cells, sd.num_faces, sd.num_ridges

def ridge(sd, r, r_ex=None, r_hat=None):
    if r_ex is not None:
        r_eval = r_ex(sd)
    if r_hat is not None:
        r_eval = r_hat

    if sd.dim == 3:
        norm_r = np.sqrt(
            np.trace((r_eval @ sps.diags(sd.cell_volumes) @ r_eval.T))
        )
        err_r = np.sqrt(
            np.trace(((r_eval - r) @ sps.diags(sd.cell_volumes) @ (r_eval - r).T))
        ) / norm_r
    else:
        mass = pg.Lagrange1("flow").assemble_mass_matrix(sd, None)
        norm_r = np.sqrt(r_eval @ mass @ r_eval.T)
        err_r = np.sqrt((r_eval - r) @ mass @ (r_eval - r).T) / norm_r
    return err_r


def face(sd, P0q, q_ex=None, q_hat=None):
    if q_ex is not None:
        q_eval = q_ex(sd)
    if q_hat is not None:
        q_eval = q_hat

    norm_q = np.sqrt(
        np.trace(q_eval @ sps.diags(sd.cell_volumes) @ q_eval.T)
    )
    err_q = np.sqrt(
        np.trace((q_eval - P0q) @ sps.diags(sd.cell_volumes) @ (q_eval - P0q).T)
    ) / norm_q
    return err_q

def cell(sd, p, p_ex=None, p_hat=None):
    if p_ex is not None:
        p_eval = p_ex(sd)
    if p_hat is not None:
        p_eval = p_hat

    norm_p = _cell_error(sd, np.zeros(sd.num_cells), p_eval)
    err_p = _cell_error(sd, p, p_eval) / norm_p
    return err_p

def cell_center(sd, p, p_ex=None, p_hat=None):
    if p_ex is not None:
        p_eval = p_ex(sd)
    if p_hat is not None:
        p_eval = p_hat

    norm_p = np.sqrt(p_eval @ sps.diags(sd.cell_volumes) @ p_eval.T)
    err_p = np.sqrt((p_eval - p) @ sps.diags(sd.cell_volumes) @ (p_eval - p).T) / norm_p

    return err_p

def order(err, h):
    err = np.asarray(err)
    h = np.asarray(h)
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
