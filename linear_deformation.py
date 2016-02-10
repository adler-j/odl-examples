# -*- coding: utf-8 -*-
"""
Example of creating an operator that acts as a linear transformation.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import odl
import numpy as np


def K(x, y, sigma):
    # Define the K matrix as symmetric gaussian
    return np.exp(-((x[0] - y[0])**2 + (x[1] - y[1])**2) / sigma**2) * np.eye(2)


def v(x, grid, alphas, sigma):
    """ Calculate the translation at point x """
    alpha1, alpha2 = alphas  # unpack translations per direction
    result = np.zeros_like(x)
    scale = 0.0  # Rescale the result by the magnitude of the kernels
    for i, (point, a1, a2) in enumerate(zip(grid.points(), alpha1, alpha2)):
        result += K(x, point, sigma).dot([a1, a2]).squeeze()
        scale += np.linalg.norm(K(x, point, sigma))
    return result / scale


class LinearDeformation(odl.Operator):
    """ A linear deformation given by:

        ``g(x) = f(x + v(x))``

    Where ``f(x)`` is the input template and ``v(x)`` is the translation at
    point ``x``. ``v(x)`` is computed using gaussian kernels with midpoints at
    ``grid``.
    """
    def __init__(self, fspace, vspace, grid, sigma):
        self.grid = grid
        self.sigma = sigma
        super().__init__(odl.ProductSpace(fspace, vspace), fspace, False)

    def _call(self, x):
        # Unpack input
        f, alphas = x
        extension = f.space.extension(f.ntuple)  # this syntax is improved in pull #276

        # Array of output values
        out_values = np.zeros(f.size)

        for i, point in enumerate(self.range.points()):
            # Calculate deformation in each point
            point += v(point, self.grid, alphas, self.sigma)

            if point in extension.domain:
                # Use extension operator of f
                out_values[i] = extension(point)
            else:
                # Zero boundary condition
                out_values[i] = 0

        return out_values

# Discretization of the space
m = 100  # Number of gridpoints for discretization
spc = odl.uniform_discr([0, 0], [1, 1], [m, m])

# deformation space
n = 5  # number of gridpoints for deformation, usually smaller than m
vspace = odl.ProductSpace(odl.uniform_discr([0, 0], [1, 1], [n, n]), 2)

# Deformation operator
deformation = LinearDeformation(spc, vspace, vspace[0].grid, sigma=0.2)

# Create input function
f = odl.util.shepp_logan(spc, True)

# Create deformation field
values = np.zeros([2, n, n])
values[0, :, :n//2] = 0.3  # movement in "x" direction
values[1, n//2, :] = 0.1   # movement in "y" direction
def_coeff = vspace.element(values)

# Show input
f.show(title='f')
def_coeff.show(title='deformation')

# Calculate deformed function
result = deformation([f, def_coeff])
result.show(title='result')
