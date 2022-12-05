import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys

sys.path.append("../../src/")

from mandel import compute_true_solutions, compute_initial_true_solutions, scale_time
from create_grid import create_grid

def main(n):

    a, b, c = 100, 10, 1
    domain = {"xmin": 0, "xmax": a, "ymin": 0, "ymax": b, "zmin": 0, "zmax": c}
    mdg = create_grid(n, 2, domain)

    elast_key = "elasticity"
    flow_key = "flow"

    # problem data
    mu = 2.475e9
    labda = 1.65e9
    alpha = 1
    c0 = 6.0606e-11
    perm = 9.869e-11
    F = 6e8

    delta = np.sqrt(1e1)
    num_time_steps = 100*50

    n_alpha = 200
    p_true, u_true = compute_true_solutions(
        [labda, mu, c0, alpha, perm], [F, a, b], n_alpha
    )

    p_0, u_0 = compute_initial_true_solutions(
        [labda, mu, c0, alpha, perm], [F, a, b]
    )

    # dscretization
    L1 = pg.Lagrange1(elast_key)
    RT0 = pg.RT0(elast_key)
    P0 = pg.PwConstants(flow_key)

    # set the bc data and source terms
    elast_bc_ess, flow_bc_ess = [], []
    for sd, data in mdg.subdomains(return_data=True):
        ess_rotation = sd.tags["domain_boundary_ridges"]
        ess_displacement = sd.tags["domain_boundary_faces"]
        elast_bc_ess.append(np.hstack((ess_rotation, ess_displacement)))

        # elasticity
        elast_param = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }

        ess_flux = sd.tags["domain_boundary_faces"]
        ess_flux[np.isclose(sd.face_centers[0, :], a)] = False
        ess_pressure = np.zeros(sd.num_cells, dtype=bool)
        flow_bc_ess.append(np.hstack((ess_flux, ess_pressure)))

        # flow
        flow_param = {
            "second_order_tensor": pp.SecondOrderTensor(perm * np.ones(sd.num_cells))
        }

        data[pp.PARAMETERS] = {elast_key: elast_param, flow_key: flow_param}
        data[pp.DISCRETIZATION_MATRICES] = {elast_key: {}, flow_key: {}}

    bc_ess = np.hstack(elast_bc_ess + flow_bc_ess).flatten()

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
    spp = sps.bmat(
        [
            [1 / mu * ridge_mass, -curl.T, None, None],
            [curl, (labda + 2*mu) * div_div, None, -alpha * div.T],
            [None, None, face_perm_mass, -delta * div.T],
            [None, alpha * div, delta * div, c0 * cell_mass],
        ],
        format="csc",
    )

    # projection matrices
    ridge_proj = pg.eval_at_cell_centers(mdg, L1)
    face_proj = pg.eval_at_cell_centers(mdg, RT0)
    cell_proj = pg.eval_at_cell_centers(mdg, P0)

    u = RT0.interpolate(sd, u_0)
    p = P0.interpolate(sd, p_0)

    rhs = np.zeros(spp.shape[0])
    bc_val = np.zeros(spp.shape[0])

    exporter = pp.Exporter(
        mdg, "sol", folder_name="sol", export_constants_separately=False
    )

    def export_sol(r, u, q, p, u_ex, p_ex, t_step):
        for sd, data in mdg.subdomains(return_data=True):
            data[pp.STATE] = {
                "r": ridge_proj * r,
                "p": cell_proj * p * a / F,
                "p_ex": cell_proj * p_ex * a / F,
                "u": (face_proj * u).reshape((3, -1), order="F") / a,
                "u_ex": (face_proj * u_ex).reshape((3, -1), order="F") / a,
                "q": (face_proj * q).reshape((3, -1), order="F"),
            }
        exporter.write_vtu([*data[pp.STATE].keys()], time_step=t_step)

    export_sol(np.zeros(sd.num_ridges), u, np.zeros(sd.num_faces), p, u, p, 0)

    for t_step in np.arange(1, num_time_steps):
        time = t_step * (delta**2)
        print(int(time), t_step, num_time_steps)

        # previous time step contribution
        rhs[-cell_dof:] = alpha * div * u + c0 * cell_mass * p

        # set the boundary condition
        u_bc = RT0.interpolate(sd, lambda x: u_true(x, time))
        bc_val[dofs[0] : dofs[1]] = u_bc

        # create the linear system
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(bc_ess, bc_val)
        x = ls.solve()

        # extract the variables
        r, u, q, p = np.split(x, dofs)

        u_at_time = RT0.interpolate(sd, lambda x: u_true(x, time))
        p_at_time = P0.interpolate(sd, lambda x: p_true(x, time))

        export_sol(r, u, q, p, u_at_time, p_at_time, t_step)

    time = np.arange(num_time_steps) * delta**2
    time_scaled = scale_time(time, [labda, mu, c0, alpha, perm], [F, a, b])
    exporter.write_pvd(time_scaled)


if __name__ == "__main__":
    main(50)
