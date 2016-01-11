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


def SplitBregmanReconstruct(A, Phi, x, rhs, la, mu,
                            iterations=1, N=1, isotropic=True, partial=None):
    """ Reconstruct with split Bregman.

    Parameters
    ----------
        A : `odl.Operator`
            Pojector
        Phi : `odl.Operator`
            Sparsifying transform
        x : ``A.domain`` element

    """
    Atf = A.adjoint(rhs)

    b = Phi.range.zero()
    d = Phi.range.zero()

    op = mu * (A.adjoint * A) + la * (Phi.adjoint * Phi)

    for i in range(iterations):
        for n in range(N):
            # Solve tomography part iteratively
            rhs = mu * Atf + la * Phi.adjoint(d-b)
            odl.solvers.conjugate_gradient(op, x, rhs, niter=1)

            # d = sign(Phi(x)+b) * max(|Phi(x)+b|-la^-1,0)
            if isotropic:
                s = Phi(x) + b
                sn = sum((Phi(x) + b)**2, A.domain.zero())
                for j in range(A.domain.ndim):
                    d[j] = sn.ufunc.add(-1.0/la).ufunc.maximum(0.0) * s[j] / sn
            else:
                s = Phi(x) + b
                d = s.ufunc.sign() * (s.ufunc.absolute().
                                      ufunc.add(-1.0/la).
                                      ufunc.maximum(0.0))

        b = b + Phi(x) - d

        if partial:
            partial.send(x)

# Problem size
n = 200

# Discrete reconstruction space
discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                     [n]*3, dtype='float32')

# Geometry
src_rad = 100
det_rad = 100
angle_intvl = odl.Interval(0, 2 * np.pi)
dparams = odl.Rectangle([-50, -50], [50, 50])
agrid = odl.uniform_sampling(angle_intvl, n)
dgrid = odl.uniform_sampling(dparams, [n]*2)
geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams,
                                         src_rad, det_rad,
                                         agrid, dgrid)

# X-ray transform
A = odl.tomo.DiscreteXrayTransform(discr_reco_space, geom,
                                   backend='astra_cuda')

# Create phantom
phantom = odl.util.shepp_logan(discr_reco_space, True)

# Adjoint currently bugged, needs to be fixed
A._adjoint *= A(phantom).inner(A(phantom)) / phantom.inner(A.adjoint(A(phantom)))

# These are tuing parameters in the algorithm
la = 30.0 / n  # Relaxation
mu = 0.03 * n  # Data fidelity

# Create projector
diff = odl.DiscreteGradient(discr_reco_space, method='forward')

# Create data
rhs = A(phantom)

# Add noise
mean = rhs.ufunc.sum() / rhs.size
rhs.ufunc.add(np.random.rand(A.range.size)*0.5*mean, out=rhs)

# Reconstruct
x = discr_reco_space.zero()
SplitBregmanReconstruct(A, diff, x, rhs, la, mu, 500, 1,
                        isotropic=True,
                        partial=odl.solvers.util.ShowPartial(clim=[0, 1]))
phantom.show()
