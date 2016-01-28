# -*- coding: utf-8 -*-
""" Solve laplace equation using ODL """

# External
import numpy as np

# Internal
import odl

n = 100

# Discrete reconstruction space
domain = odl.uniform_discr([0, 0], [1, 1], [n, n])

# Define right hand side
x, y = domain.grid.meshgrid(squeeze=True)
rhs_arr = np.zeros([n, n])
rhs_arr[0, :] = 0.25 * np.sin(np.pi*y)
rhs_arr[-1, :] = 1.00 * np.sin(np.pi*y)
rhs_arr[:, 0] = 0.50 * np.sin(np.pi*x)
rhs_arr[:, -1] = 0.50 * np.sin(np.pi*x)
rhs_arr *= n**2
rhs = domain.element(rhs_arr)

# Define operator
laplacian = -odl.Laplacian(domain)

# Solve with conjugate gradient
x = domain.zero()
odl.solvers.conjugate_gradient(laplacian, x, rhs, niter=300,
                               partial=odl.solvers.ShowPartial())
