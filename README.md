# Mixed and Multipoint Finite Element Methods for Rotation-based Poroelasticity
### Wietse M. Boon, Alessio Fumagalli, and Anna Scotti

The [examples](./examples/) folder contains the source code for replicating the three test cases. See [arXiv pre-print]().<br>

# Abstract
This work proposes a mixed finite element method for the Biot poroelasticity equations that employs the lowest-order Raviart-Thomas finite element space for the solid displacement and piecewise constants for the fluid pressure. The method is based on the formulation of linearized elasticity as a weighted vector Laplace problem. By introducing the solid rotation and fluid flux as auxiliary variables, we form a four-field formulation of the Biot system, which is discretized using conforming mixed finite element spaces. The auxiliary variables are subsequently removed from the system in a local hybridization technique to obtain a multipoint rotation-flux mixed finite element method. Stability and convergence of the four-field and multipoint mixed finite element methods are shown in terms of weighted norms, which additionally leads to parameter-robust preconditioners. Numerical experiments confirm the theoretical results

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv pre-print]().

# PorePy and PyGeoN version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and [PyGeoN](https://github.com/compgeo-mox/pygeon) and might revert them.
Newer versions of may not be compatible with this repository.<br>
PorePy valid commit: b3d8441065ac2c56a5f548a34c57a49984865ad0 <br>
PyGeoN valid tag: v0.2.0

# License
See [license](./LICENSE).
