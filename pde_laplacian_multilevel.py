# -*- coding: utf-8 -*-
""" Solve laplace equation using ODL """

import numpy as np
import odl

space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))

def boundary_values(point):
    x, y = point
    result = 0.25 * np.sin(np.pi*y) * (x == 0)
    result += 1.00 * np.sin(np.pi*y) * (x == 1)
    result += 0.50 * np.sin(np.pi*x) * (y == 0)
    result += 0.50 * np.sin(np.pi*x) * (y == 1)
    return result

n_last = 1
for n in [5, 50, 500]:
    # Discrete reconstruction space
    domain = odl.uniform_discr_fromspace(space, [n, n],
                                         nodes_on_bdry=True, interp='linear')

    # Define right hand side
    rhs = domain.element(boundary_values)

    # Define operator
    laplacian = odl.Laplacian(domain) * (-1.0 / n**2)

    if n_last==1:
        # Pick initial guess
        vec = domain.zero()
    else:
        # Extend last value if possible
        extension = vec.space.extension(vec.ntuple)
        vec = domain.element(extension)

    # Solve with conjugate gradient
    odl.solvers.conjugate_gradient(laplacian, vec, rhs, niter=n,
                                   partial=odl.solvers.ShowPartial())
    n_last = n

