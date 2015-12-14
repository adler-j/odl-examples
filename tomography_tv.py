# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:34:27 2015

@author: JonasAdler
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super, range

# External
from skimage.transform import radon, iradon
import numpy as np

# Internal
import odl


class ForwardProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = ran.grid.meshgrid()[1][0] * 180 / np.pi
        super().__init__(dom, ran, True)

    def _call(self, x):
        scale = self.range.grid.stride[0] * d.shape[0] / (d.shape[0] - 1.0)
        return self.range.element(radon(x.asarray(), self.theta)) * scale

    @property
    def adjoint(self):
        return BackProjector(self.range, self.domain)


class BackProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = dom.grid.meshgrid()[1][0] * 180 / np.pi
        self.npoint = ran.grid.shape[0]
        super().__init__(dom, ran, True)

    def _call(self, x):
        return self.range.element(iradon(x.asarray(), self.theta, self.npoint,
                                         filter=None))

    @property
    def adjoint(self):
        return ForwardProjector(self.range, self.domain)


def TVreconstruct2D(A, x, rhs, la, mu, iterations=1, N=2):
    diff = odl.DiscreteGradient(x.space, method='forward')

    Atf = A.adjoint(rhs)

    b = diff.range.zero()
    d = diff.range.zero()

    op = mu * (A.adjoint * A) + la * (diff.adjoint * diff)

    for i in odl.util.ProgressRange("denoising", iterations):
        for n in range(N):
            # Solve tomography part iteratively
            rhs = mu * Atf + la * diff.adjoint(d-b)
            odl.solvers.conjugate_gradient(op, x, rhs, niter=1)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            s = diff(x) + b
            d = s.ufunc.sign() * (s.ufunc.absolute().
                                  ufunc.add(-1.0/la).
                                  ufunc.maximum(0.0))

        b = b + diff(x) - d

        x.show()

n = 100

# Create spaces
d = odl.uniform_discr([0, 0], [1, 1], [n, n])
ran = odl.uniform_discr([0, 0], [1, np.pi], [np.ceil(np.sqrt(2) * n), n])

# Create phantom
phantom = odl.util.shepp_logan(d)
phantom.show()

# These are tuing parameters in the algorithm
la = 2.0 / n # Relaxation
mu = 500.0 * n # Data fidelity

# Create projector
A = ForwardProjector(d, ran)

# Create data
rhs = A(phantom)
rhs.show()

# Add noise
rhs.ufunc.add(np.random.rand(ran.size)*0.05, out=rhs)
rhs.show()

# Reconstruct
x = d.zero()
TVreconstruct2D(A, x, rhs, la, mu, 50, 1)
phantom.show()
