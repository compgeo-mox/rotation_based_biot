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

class minres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            #print('iter %3i\trk = %s' % (self.niter, str(rk)))
            print('iter %3i' % self.niter)

def main(n, mu, labda):
    print("Perform simulation for", n, mu, labda)

    mdg = create_grid(n)
    keyword="flow"

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

    # construct the matrices
    ridge_mass = pg.ridge_mass(mdg)
    cell_mass = pg.cell_mass(mdg)
    face_mass = pg.face_mass(mdg)

    curl = face_mass * pg.curl(mdg)
    div = cell_mass * pg.div(mdg)

    div_div = pg.div(mdg).T * cell_mass * pg.div(mdg)

    # get the degrees of freedom for each variable
    face_dof, ridge_dof = curl.shape
    dofs = np.cumsum([ridge_dof])

    # assemble the problem
    mat = sps.bmat([
        [2/mu*ridge_mass,                -curl.T],
        [           -curl, -(labda + mu)*div_div]
        ], format="csc")

    # assemble the right-hand side
    rhs = np.hstack(bc_val)
    rhs[dofs[0]:] += face_mass * np.hstack(v_source)

    L1 = pg.Lagrange1(keyword)
    RT0 = pg.RT0(keyword)

    p1 = 2/mu*(ridge_mass + L1.assemble_stiff_matrix(sd, data))
    p2 = mu/2*face_mass + (labda + mu) * RT0.assemble_stiff_matrix(sd, data)

    precond = sps.block_diag([p1, p2], format="csc")

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

    A_0, _, _ = ls.reduce_system()
    l_M = sps.linalg.eigsh(A_0, k=1, M=precond_0, which="LM", return_eigenvectors=False, tol=1e-4)
    l_m = sps.linalg.eigsh(A_0, k=1, M=precond_0, which="SM", return_eigenvectors=False, tol=1e-4)

    return counter.niter, l_M[0], l_m[0]

if __name__ == "__main__":

    n_val = 2 ** np.arange(4, 9)
    mu_val = np.power(10., np.arange(-4, 5, 2))
    labda_val = np.power(10., np.arange(-4, 5, 2))

    results = np.array([(n, mu, labda,
                        *main(n, mu, labda))
                         for n in n_val \
                         for mu in mu_val \
                         for labda in labda_val])

    np.savetxt("iterations.txt", results)
