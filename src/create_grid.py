import porepy as pp
import pygeon as pg

def create_grid(n, dim, domain=None):
    # make the grid
    if domain is None:
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    if dim == 2:
        network = pp.FractureNetwork2d(domain=domain)
    elif dim == 3:
        network = pp.FractureNetwork3d(domain=domain)

    mesh_size = (domain["xmax"] - domain["xmin"])/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

