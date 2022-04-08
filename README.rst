JFPI
====

Julia Firedrake Projection Interface (JFPI) is an experimental library to transfer data from particles to finite element function spaces. For example a charge distribution stored as particles can be projected onto a scalar function. Finite element functions can also be evaluated at particle locations.

Assumes that the domain is a structured Cartesian mesh, e.g. 2D quad mesh. Projections and evaluations are performed through DG function spaces with a user choice of polynomial order.
