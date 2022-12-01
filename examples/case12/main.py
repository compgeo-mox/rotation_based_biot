import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
import error

def elast_vector_source(pt):
    x, y, z = pt

    first = -16*x**2*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1) + 8*x**2*y**2*z**2*(x - 1)*(y - 1)*(z - 1)**2 - 16*x**2*y**2*z**2*(y - 1)**2*(z - 1)**2 - 16*x**2*y**2*z*(x - 1)*(y - 1)**2*(z - 1)**2 + 8*x**2*y*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 - 16*x*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1)**2 - 64*x*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 - 16*x*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 + x*y**2*z*(x - 1)*(y - 1)**2*(z - 1)*(2.0*x*z - 8.0*x*(x - 1) + 2.0*x*(z - 1) + 2.0*z*(x - 1) + 2.0*(x - 1)*(z - 1)) - x*y**2*z*(x - 1)*(y - 1)**2*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) - x*y**2*(x - 1)*(y - 1)**2*(z - 1)*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1)) + 8*x*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - x*y*z**2*(x - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 2.0*x*y*(y - 1) + 12.0*x*(x - 1)*(y - 1) + 1.0*x*(y - 1)**2 + 2.0*y*(x - 1)*(y - 1) + 1.0*(x - 1)*(y - 1)**2) - x*y*z*(y - 1)*(z - 1) - x*z**2*(x - 1)*(y - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1)) - 16*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - y*z*(x - 1)*(y - 1)*(z - 1)
    second = -16*x**2*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1) + 4*x**2*y**2*z**2*(x - 1)**2*(z - 1)**2 - 32*x**2*y**2*z**2*(x - 1)*(y - 1)*(z - 1)**2 - 16*x**2*y**2*z*(x - 1)**2*(y - 1)*(z - 1)**2 - 16*x**2*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 16*x**2*y*z**2*(x - 1)**2*(y - 1)*(z - 1)**2 - 32*x**2*y*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 - 16*x**2*y*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 + 2.0*x**2*y*z*(x - 1)**2*(y - 1)*(z - 1)*(y*z + y*(y - 1) + y*(z - 1) + z*(y - 1) + (y - 1)*(z - 1)) + x**2*y*z*(x - 1)**2*(y - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + x**2*y*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) + 4*x**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - 32*x*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1)**2 - 32*x*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 + x*y*z**2*(y - 1)*(z - 1)**2*(8.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 8.0*x*(x - 1)*(y - 1) + 4.0*y*(x - 1)**2 + 3.0*y*(x - 1)*(y - 1) + 4.0*(x - 1)**2*(y - 1)) - x*y*z*(x - 1)*(z - 1) - x*z*(x - 1)*(y - 1)*(z - 1) + y*z**2*(x - 1)*(y - 1)*(z - 1)**2*(4.0*x*y*(x - 1) + 1.0*x*y*(y - 1) + 4.0*x*(x - 1)*(y - 1) + 1.0*y*(x - 1)*(y - 1))
    third = -8*x**2*y**2*z**2*(x - 1)**2*(y - 1)**2 + 8*x**2*y**2*z**2*(x - 1)**2*(y - 1)*(z - 1) - 32*x**2*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1) - 32*x**2*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x**2*y**2*z*(x - 1)**2*(y - 1)*(z - 1)**2 - 32*x**2*y**2*z*(x - 1)*(y - 1)**2*(z - 1)**2 - 8*x**2*y**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 + 8*x**2*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x**2*y*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 - x**2*y*z*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z + 1.0*y*(z - 1) + 1.0*z*(y - 1) + 4.0*z*(z - 1) + 1.0*(y - 1)*(z - 1)) - x**2*y*z*(x - 1)**2*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) - x**2*z*(x - 1)**2*(y - 1)*(z - 1)*(1.0*y*z*(y - 1) + 2.0*y*z*(z - 1) + 1.0*y*(y - 1)*(z - 1) + 2.0*z*(y - 1)*(z - 1)) - 32*x*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1) - 32*x*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 + x*y**2*z*(y - 1)**2*(z - 1)*(8.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 8.0*x*(x - 1)*(z - 1) + 4.0*z*(x - 1)**2 - 6.0*z*(x - 1)*(z - 1) + 4.0*(x - 1)**2*(z - 1)) - x*y*z*(x - 1)*(y - 1) - x*y*(x - 1)*(y - 1)*(z - 1) + y**2*z*(x - 1)*(y - 1)**2*(z - 1)*(4.0*x*z*(x - 1) - 2.0*x*z*(z - 1) + 4.0*x*(x - 1)*(z - 1) - 2.0*z*(x - 1)*(z - 1))

    return np.vstack((first, second, third))

def flow_vector_source(pt):
    x, y, z = pt

    first = -x*y*z*(y - 1)*(z - 1) - y*z*(x - 1)*(y - 1)*(z - 1) + np.sin(2*x*np.pi)*np.sin(2*y*np.pi)*np.sin(2*z*np.pi)
    second = x*z*(x - 1)*(z - 1)*(-y*(y - 1) - 2*y + 1)
    third = -y*(y - 1)*(x*z*(x - 1) + x*(x - 1)*(z - 1) + np.sin(2*x*np.pi)*np.sin(2*z*np.pi))

    return np.vstack((first, second, third))

def flow_scalar_source(pt):
    x, y, z = pt

    return 2*x**2*y**2*z**2*(1 - y)*(x - 1)**2*(z - 1)**2 + 4*x**2*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1) + 8*x**2*y**2*z**2*(x - 1)*(y - 1)**2*(z - 1)**2 + 4*x**2*y**2*z*(x - 1)**2*(y - 1)**2*(z - 1)**2 - 2*x**2*y*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 + 8*x*y**2*z**2*(x - 1)**2*(y - 1)**2*(z - 1)**2 - x*y*z*(x - 1)*(y - 1)*(z - 1) - x*y*z*(x - 1)*(z - 1) - x*z*(x - 1)*(y - 1)*(z - 1) - 2*y*np.pi*(y - 1)*np.sin(2*x*np.pi)*np.cos(2*z*np.pi) + 2*np.pi*np.sin(2*y*np.pi)*np.sin(2*z*np.pi)*np.cos(2*x*np.pi)

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

class minres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            #print('iter %3i\trk = %s' % (self.niter, str(rk)))
            print('iter %3i' % self.niter)

def main(n, mu, labda, k, alpha, c0, delta):
    print("Perform simulation for", n, mu, labda, k, alpha, c0, delta)

    mdg = create_grid(n)
    elast_key = "elasticity"
    flow_key = "flow"

    # create the preconditioner
    N0 = pg.Nedelec0(elast_key)
    RT0 = pg.RT0(elast_key)
    P0 = pg.PwConstants(flow_key)

    # set the bc data and source terms
    elast_bc_ess, elast_v_source = [], []
    flow_bc_ess, flow_v_source, flow_s_source = [], [], []
    for sd, data in mdg.subdomains(return_data=True):

        # elasticity
        elast_param = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }

        ess_rotation = sd.tags["domain_boundary_ridges"]
        ess_displacement = sd.tags["domain_boundary_faces"]
        elast_bc_ess.append(np.hstack((ess_rotation, ess_displacement)))

        elast_v_source.append(RT0.interpolate(sd, elast_vector_source))

        # flow
        flow_param = {
            "second_order_tensor": pp.SecondOrderTensor(k*np.ones(sd.num_cells))
        }

        ess_flux = np.zeros(sd.num_faces, dtype=bool)
        ess_pressure = np.zeros(sd.num_cells, dtype=bool)

        flow_bc_ess.append(np.hstack((ess_flux, ess_pressure)))

        flow_v_source.append(RT0.interpolate(sd, flow_vector_source))
        flow_s_source.append(P0.interpolate(sd, flow_scalar_source))

        data[pp.PARAMETERS] = {elast_key: elast_param, flow_key: flow_param}
        data[pp.DISCRETIZATION_MATRICES] = {elast_key: {}, flow_key: {}}

    bc_ess = np.hstack((elast_bc_ess, flow_bc_ess)).flatten()

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
    mat = sps.bmat([
        [2/mu*ridge_mass,               -curl.T,            None,         None],
        [          -curl, -(labda + mu)*div_div,            None,  alpha*div.T],
        [           None,                  None, -face_perm_mass,  delta*div.T],
        [           None,             alpha*div,       delta*div, c0*cell_mass]
        ], format="csc")

    # assemble the right-hand side
    rhs = np.zeros(mat.shape[0])
    # data from the elastic problem
    rhs[dofs[0]:dofs[1]] += face_mass * np.hstack(elast_v_source)
    # data from the flow problem
    rhs[dofs[1]:dofs[2]] += face_mass * np.hstack(flow_v_source)
    rhs[dofs[2]:] += cell_mass * np.hstack(flow_s_source)

    face_stiff = RT0.assemble_stiff_matrix(sd, data)
    eta = alpha**2 / (labda + mu) + delta**2 * k

    p1 = 2/mu*(ridge_mass + N0.assemble_stiff_matrix(sd, data))
    p2 = mu/2*face_mass + (labda + mu) * face_stiff
    p3 = 1/k*face_mass + (delta**2) / (eta + c0) * face_stiff
    p4 = (eta + c0) * cell_mass

    precond = sps.block_diag([p1, p2, p3, p4], format="csc")

    ls_precond = pg.LinearSystem(precond)
    ls_precond.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    precond_0, _, _ = ls_precond.reduce_system()
    apply_precond = lambda x: sps.linalg.spsolve(precond_0, x)
    M = sps.linalg.LinearOperator(precond_0.shape, matvec=apply_precond)

    counter = minres_counter()
    solver = lambda A, b: sps.linalg.minres(A, b, M=M, callback=counter)[0]

    # solve the problem
    ls = pg.LinearSystem(mat, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve(solver=solver)

    #A_0, _, _ = ls.reduce_system()
    #l_M = sps.linalg.eigsh(A_0, k=1, M=precond_0, which="LM", return_eigenvectors=False, tol=1e-4)
    #l_m = sps.linalg.eigsh(A_0, k=1, M=precond_0, which="SM", return_eigenvectors=False, tol=1e-4)

    return counter.niter#, l_M[0], l_m[0]

if __name__ == "__main__":

    n_val =  [19, 23] #[7, 11, 15]# , 19, 23]
    mu_val = np.array([1e-4, 1, 1e4])
    labda_val = np.array([1e-4, 1, 1e4])
    k_val = np.array([1e-4, 1, 1e4])
    alpha_val = np.array([1])
    c0_val = np.array([0, 10, 100])
    delta_val = np.array([1e-4, 1e-2, 1])

    results = np.array([(n, mu, labda, k, alpha, c0, delta,
                         main(n, mu, labda, k, alpha, c0, delta))
                          for n in n_val \
                          for mu in mu_val \
                          for labda in labda_val \
                          for k in k_val \
                          for alpha in alpha_val \
                          for c0 in c0_val \
                          for delta in delta_val])

    np.savetxt("iterations1.txt", results)
