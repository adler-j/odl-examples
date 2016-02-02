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


def single_cg_iter(op, x, rhs):
    r = op(x)
    r.lincomb(1, rhs, -1, r)       # r = rhs - A x
    sqnorm_r_old = r.norm() ** 2  # Only recalculate norm after update
    d = op(r)  # d = A r
    inner_p_d = r.inner(d)
    alpha = sqnorm_r_old / inner_p_d
    x.lincomb(1, x, alpha, r)            # x = x + alpha*r


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
            single_cg_iter(op, x, rhs) #odl.solvers.conjugate_gradient(op, x, rhs, niter=1)

            s = Phi(x) + b
            if isotropic:
                sn = sum(s**2, A.domain.zero())
                for j in range(A.domain.ndim):
                    d[j] = sn.ufunc.add(-1.0/la).ufunc.maximum(0.0) * s[j] / sn
            else:
                # d = sign(Phi(x)+b) * max(|Phi(x)+b|-la^-1,0)
                d = s.ufunc.sign() * (s.ufunc.absolute().
                                      ufunc.add(-1.0/la).
                                      ufunc.maximum(0.0))

        b = b + Phi(x) - d

        if partial:
            partial(x)

# Problem size
n = 200

# Discrete reconstruction space
discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                     [n]*3, dtype='float32')

# Geometry
src_rad = 1000
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

# Create data
rhs = A(phantom)

# Adjoint currently bugged, needs to be fixed
A._adjoint *= rhs.inner(rhs) / phantom.inner(A.adjoint(rhs))

# These are tuing parameters in the algorithm
la = 30.0 / n  # Relaxation
mu = 0.1 * n  # Data fidelity

# Create projector
diff = odl.Gradient(discr_reco_space, method='forward')


# Add noise
mean = rhs.ufunc.sum() / rhs.size
rhs.ufunc.add(np.random.rand(A.range.size)*1.0*mean, out=rhs)

partial = (odl.solvers.util.ShowPartial(clim=[-0.1, 1.1]) &
           odl.solvers.util.ShowPartial(indices=np.s_[:, n//2, n//2]) &
           odl.solvers.util.PrintIterationPartial())

# Reconstruct
x = discr_reco_space.zero()
SplitBregmanReconstruct(A, diff, x, rhs, la, mu, 500, 1,
                        isotropic=True, partial=partial)
phantom.show()
