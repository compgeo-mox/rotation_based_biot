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
    BDM1 = pg.BDM1(flow_key)

    # set the bc data and source terms
    ess_displacement, ess_pressure = [], []
    ess_rotation, ess_flux = [], []
    for sd, data in mdg.subdomains(return_data=True):
        ess_displacement.append(sd.tags["domain_boundary_faces"])
        ess_rotation.append(sd.tags["domain_boundary_ridges"])

        # elasticity
        elast_param = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }

        ess_pressure.append(np.zeros(sd.num_cells, dtype=bool))

        bc_flux = sd.tags["domain_boundary_faces"].copy()
        bc_flux[np.isclose(sd.face_centers[0, :], a)] = False
        ess_flux.append(bc_flux)
        ess_flux.append(bc_flux)

        # flow
        flow_param = {
            "second_order_tensor": pp.SecondOrderTensor(perm * np.ones(sd.num_cells))
        }

        data[pp.PARAMETERS] = {elast_key: elast_param, flow_key: flow_param}
        data[pp.DISCRETIZATION_MATRICES] = {elast_key: {}, flow_key: {}}

    ess_rotation = np.hstack(ess_rotation)
    ess_flux = np.hstack(ess_flux)
    ess_displacement = np.hstack(ess_displacement)
    ess_pressure = np.hstack(ess_pressure)

    bc_ess = np.hstack((ess_displacement, ess_pressure))

    # construct the matrices
    lumped_ridge_mass = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, None)
    cell_mass = pg.cell_mass(mdg)
    face_mass = pg.face_mass(mdg, keyword=elast_key)

    lumped_face_perm_mass = 1/perm * BDM1.assemble_lumped_matrix(sd, None)
    div_bdm1 = cell_mass * BDM1.assemble_diff_matrix(sd)

    curl = face_mass * pg.curl(mdg)
    div = cell_mass * pg.div(mdg)
    div_div = pg.div(mdg).T * cell_mass * pg.div(mdg)

    # assemble the Schur complement for the elastic problem
    ls = pg.LinearSystem(1/mu*lumped_ridge_mass, curl.T.tocsc())
    ls.flag_ess_bc(ess_rotation, np.zeros_like(ess_rotation))
    elast_sc = ls.solve()

    # assemble the reduced matrix for the elastic problem
    elast_mat = curl * elast_sc + (labda+2*mu)*div_div

    # assemble the Schur complement for the flow problem
    ls = pg.LinearSystem(lumped_face_perm_mass, delta * div_bdm1.T.tocsc())
    ls.flag_ess_bc(ess_flux, np.zeros_like(ess_flux))
    flow_sc = ls.solve()

    # assemble the reduced matrix for the flow problem
    flow_mat = delta * div_bdm1 * flow_sc + c0 * cell_mass

    # get the degrees of freedom for each variable
    cell_dof, face_dof = div.shape
    dofs = np.cumsum([face_dof])

    # assemble the saddle point problem
    spp = sps.bmat([
        [elast_mat, -alpha*div.T],
        [alpha*div,     flow_mat]
        ], format="csc")

    # projection matrices
    face_proj = pg.eval_at_cell_centers(mdg, RT0)
    cell_proj = pg.eval_at_cell_centers(mdg, P0)

    u = RT0.interpolate(sd, u_0)
    p = P0.interpolate(sd, p_0)

    rhs = np.zeros(spp.shape[0])
    bc_val = np.zeros(spp.shape[0])

    exporter = pp.Exporter(
        mdg, "sol", folder_name="sol", export_constants_separately=False
    )

    def export_sol(u, p, u_ex, p_ex, t_step):
        for sd, data in mdg.subdomains(return_data=True):
            data[pp.STATE] = {
                "p": cell_proj * p * a / F,
                "p_ex": cell_proj * p_ex * a / F,
                "u": (face_proj * u).reshape((3, -1), order="F") / a,
                "u_ex": (face_proj * u_ex).reshape((3, -1), order="F") / a,
            }
        exporter.write_vtu([*data[pp.STATE].keys()], time_step=t_step)

    export_sol(u, p, u, p, 0)

    for t_step in np.arange(1, num_time_steps):
        time = t_step * (delta**2)
        print(int(time), t_step, num_time_steps)

        # previous time step contribution
        rhs[-cell_dof:] = alpha * div * u + c0 * cell_mass * p

        # set the boundary condition
        u_bc = RT0.interpolate(sd, lambda x: u_true(x, time))
        bc_val[:dofs[0]] = u_bc

        # create the linear system
        ls = pg.LinearSystem(spp, rhs)
        ls.flag_ess_bc(bc_ess, bc_val)
        x = ls.solve()

        # extract the variables
        u, p = np.split(x, dofs)

        u_at_time = RT0.interpolate(sd, lambda x: u_true(x, time))
        p_at_time = P0.interpolate(sd, lambda x: p_true(x, time))

        export_sol(u, p, u_at_time, p_at_time, t_step)

    time = np.arange(num_time_steps) * delta**2
    time_scaled = scale_time(time, [labda, mu, c0, alpha, perm], [F, a, b])
    exporter.write_pvd(time_scaled)


if __name__ == "__main__":
    main(50)
