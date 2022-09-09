from sympy import cos, sin, diff, pi
from sympy.physics.vector import ReferenceFrame, curl, gradient, divergence
from sympy.printing.pycode import NumPyPrinter

def to_numpy(expr, R, i):
    return NumPyPrinter().doprint(expr.to_matrix(R)[i]).replace("numpy", "np").replace("R_", "")

def compute_source(u, r, R):
    """
    Given a displacement u compute the body force
    """

    div_u = divergence(u, R)
    grad_div_u = gradient(div_u, R)
    curl_r = curl(r, R)

    return (curl_r - 2*grad_div_u).simplify()


## -------------------------------------------------------------------##
def two_dim(R):
    """
    The two-dimensional polynomial case
    """

    x, y, z = R.varlist
    i, j, k = (R.x, R.y, R.z)

    u = i * (4*x**2*(1-x)**2*y**2*(1-y)**2)\
      + j * (-x**2*(1-x)**2*y**2*(1-y)**2)

    r = curl(0.5*u, R).simplify()
    g = compute_source(u, r, R)

    print("velocity\n", to_numpy(u, R, 0))
    print("velocity\n", to_numpy(u, R, 1))

    print("rotation\n", to_numpy(r, R, 2))

    print("source\n", to_numpy(g, R, 0))
    print("source\n", to_numpy(g, R, 1))


## -------------------------------------------------------------------##

if __name__ == "__main__":
    R = ReferenceFrame("R")

    two_dim(R)
