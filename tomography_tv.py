# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:34:27 2015

@author: JonasAdler
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

# Internal
import odl
from tomography_helper import ForwardProjector


def SplitBregmanReconstruct(A, Phi, x, rhs, la, mu, iterations=1, N=1):
    """ Reconstruct with split Bregman.

    Parameters
    ----------
        A : `odl.Operator`
            Pojector
        Phi : `odl.Operator`
            Sparsifying transform
        x : ``A.domain`` element

    """
    isotropic = True
    Atf = A.adjoint(rhs)

    b = Phi.range.zero()
    d = Phi.range.zero()

    op = mu * (A.adjoint * A) + la * (Phi.adjoint * Phi)

    fig = None
    for i in range(iterations):
        for n in range(N):
            # Solve tomography part iteratively
            rhs = mu * Atf + la * Phi.adjoint(d-b)
            odl.solvers.conjugate_gradient(op, x, rhs, niter=1)

            # d = sign(Phi(x)+b) * max(|Phi(x)+b|-la^-1,0)
            if isotropic:
                s = Phi(x) + b
                sn = sum((Phi(x) + b)**2, A.domain.zero())
                for i in range(A.domain.ndim):
                    d[i] = sn.ufunc.add(-1.0/la).ufunc.maximum(0.0) * s[i] / sn
            else:
                s = Phi(x) + b
                d = s.ufunc.sign() * (s.ufunc.absolute().
                                      ufunc.add(-1.0/la).
                                      ufunc.maximum(0.0))

        b = b + Phi(x) - d

        fig = x.show(clim=[0, 1], fig=fig)

n = 200

# Create spaces
d = odl.uniform_discr([0, 0], [1, 1], [n, n])
ran = odl.uniform_discr([0, 0], [1, np.pi], [np.ceil(np.sqrt(2) * n), n])

# Create phantom
phantom = odl.util.shepp_logan(d)

# These are tuing parameters in the algorithm
la = 3.0 / n  # Relaxation
mu = 200.0 * n  # Data fidelity

# Create projector
diff = odl.DiscreteGradient(d, method='forward')
A = ForwardProjector(d, ran)

# Create data
rhs = A(phantom)

# Add noise
rhs.ufunc.add(np.random.rand(ran.size)*0.02, out=rhs)

# Reconstruct
x = d.zero()
SplitBregmanReconstruct(A, diff, x, rhs, la, mu, 500, 1)
phantom.show()
