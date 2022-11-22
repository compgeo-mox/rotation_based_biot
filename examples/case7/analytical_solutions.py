from sympy import cos, sin, diff, pi
from sympy.physics.vector import ReferenceFrame, curl, gradient, divergence
from sympy.printing.pycode import NumPyPrinter

mu = 1
labda = 1
delta = 1
alpha = 1
inv_k = 1
c0 = 1

def to_numpy_vector(expr, R, i):
    return NumPyPrinter().doprint(expr.to_matrix(R)[i]).replace("numpy", "np").replace("R_", "")

def to_numpy_scalar(expr, R):
    return NumPyPrinter().doprint(expr).replace("numpy", "np").replace("R_", "")

def elast_scalar_source(u, r, p, R):
    """
    Given a displacement u compute the body force
    """

    div_u = divergence(u, R)
    grad_div_u = gradient(div_u, R)
    curl_r = curl(r, R)
    grad_p = gradient(p, R)

    return (curl_r - (labda + mu)*grad_div_u + alpha*grad_p).simplify()

def flow_vector_source(q, p, R):
    grad_p = gradient(p, R)

    return (inv_k * q + delta*grad_p).simplify()

def flow_scalar_source(u, q, p, R):
    div_u = divergence(u, R)
    div_q = divergence(q, R)

    return (alpha*div_u + delta*div_q + c0*p).simplify()

##-------------------------------------------------------------------##

def two_dim(R):
    """
    The two-dimensional polynomial case
    """

    x, y, z = R.varlist
    i, j, k = (R.x, R.y, R.z)

    u = i * (4*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2)\
      + j * (-x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2)\
      + k * (2*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2)

    print("displacement\n", to_numpy_vector(u, R, 0))
    print("displacement\n", to_numpy_vector(u, R, 1))
    print("displacement\n", to_numpy_vector(u, R, 2))

    r = curl(0.5*mu*u, R).simplify()
    print("rotation\n", to_numpy_vector(r, R, 0))
    print("rotation\n", to_numpy_vector(r, R, 1))
    print("rotation\n", to_numpy_vector(r, R, 2))

    p = x*(1-x)*y*(1-y)*z*(1-z)
    print("pressure\n", to_numpy_scalar(p, R))

    q = i * (sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z))\
      + j * (x*(1-x)*y*(1-y)*z*(1-z))\
      + k * (sin(2*pi*x)*y*(1-y)*sin(2*pi*z))
    print("velocity\n", to_numpy_vector(q, R, 0))
    print("velocity\n", to_numpy_vector(q, R, 1))
    print("velocity\n", to_numpy_vector(q, R, 2))

    g = elast_scalar_source(u, r, p, R)
    print("elastic scalar source\n", to_numpy_vector(g, R, 0))
    print("elastic scalar source\n", to_numpy_vector(g, R, 1))
    print("elastic scalar source\n", to_numpy_vector(g, R, 2))

    v = flow_vector_source(q, p, R)
    print("flow vector source\n", to_numpy_vector(v, R, 0))
    print("flow vector source\n", to_numpy_vector(v, R, 1))
    print("flow vector source\n", to_numpy_vector(v, R, 2))

    w = flow_scalar_source(u, q, p, R)
    print("flow scalar source\n", to_numpy_scalar(w, R))

    curl_r = curl(r, R).simplify()
    print("curl_r\n", to_numpy_vector(curl_r, R, 0))
    print("curl_r\n", to_numpy_vector(curl_r, R, 1))
    print("curl_r\n", to_numpy_vector(curl_r, R, 2))

    div_u = divergence(u, R).simplify()
    print("div_u\n", to_numpy_scalar(div_u, R))

    div_q = divergence(q, R).simplify()
    print("div_q\n", to_numpy_scalar(div_q, R))

##-------------------------------------------------------------------##

if __name__ == "__main__":
    R = ReferenceFrame("R")

    two_dim(R)
