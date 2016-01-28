# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:34:27 2015

@author: JonasAdler
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np

# Internal
import odl
cgn = odl.solvers.conjugate_gradient_normal


def mu0(E):
    return E ** 3


def mu1(E):
    return E ** 2  # np.float64(E<=0.7)


class SpectralDetector(odl.Operator):
    """ A detector that takes two material wise projections and returns
    the corresponding projection image.
    """
    def __init__(self, space, energies, spectrum):
        self.energies = np.array(energies)
        self.spectrum = np.array(spectrum)
        self.spectrum = self.spectrum / self.spectrum.sum()
        super().__init__(odl.ProductSpace(space, 2),
                         space,
                         False)

    def _call(self, x):
        result = self.range.zero()
        tmp = self.range.element()
        for I, E in zip(self.spectrum, self.energies):
            tmp.lincomb(-mu0(E), x[0], -mu1(E), x[1])
            tmp.ufunc.exp(out=tmp)
            result.lincomb(1, result, I, tmp)

        # -log(result)
        result.ufunc.log(out=result)
        result.lincomb(-1, result)
        return result

    def derivative(self, x):
        Ax = self(x)
        tmp = self.range.element()

        scale0 = self.range.zero()
        scale1 = self.range.zero()
        for I, E in zip(self.spectrum, self.energies):
            tmp.lincomb(1, Ax, -mu0(E), x[0])
            tmp.lincomb(1, tmp, -mu1(E), x[1])
            tmp.ufunc.exp(out=tmp)
            scale0.lincomb(1, scale0, I * mu0(E), tmp)
            scale1.lincomb(1, scale1, I * mu1(E), tmp)

        return odl.ReductionOperator(odl.MultiplyOperator(scale0),
                                     odl.MultiplyOperator(scale1))


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
proj = odl.tomo.DiscreteXrayTransform(discr_reco_space, geom,
                                      backend='astra_cuda')


# Create phantom
phantom0 = odl.util.shepp_logan(discr_reco_space, True)
phantom1 = odl.util.derenzo_sources(discr_reco_space)

# Adjoint currently bugged, needs to be fixed
proj._adjoint *= proj(phantom0).inner(proj(phantom0)) / phantom0.inner(proj.adjoint(proj(phantom0)))

# Not scaled correctly
proj = proj/5.0

# Create product space
proj_op = odl.diagonal_operator([proj, proj])

energies = np.linspace(0.5, 1.0, 5)
spectrum_low = np.exp(-((energies-0.5) * 2)**2)
spectrum_high = np.exp(-((energies-1.0) * 2)**2)
A_low = SpectralDetector(proj.range, energies, spectrum_low)
A_high = SpectralDetector(proj.range, energies, spectrum_high)

detector_op = odl.BroadcastOperator(A_low, A_high)

# Compose operators
spectral_proj = detector_op * proj_op

# Create data
phantom = spectral_proj.domain.element([phantom0, phantom1])
proj_op(phantom).show(title='materials', clim=[0, 5])
projections = spectral_proj(phantom)
projections.show(title='spectral', clim=[0, 5])

# Reconstruct with big op
if 0:
    partial = (odl.solvers.util.ShowPartial(indices=np.s_[0, :, :, :, n//2], clim=[0, 5]) &
               odl.solvers.util.ShowPartial(indices=np.s_[1, :, n//2, :, n//2]) &
               odl.solvers.util.ShowPartial(indices=np.s_[1, :, :, :, n//2], clim=[0, 1]) &
               odl.solvers.util.PrintIterationPartial())

    bigop = odl.ProductSpaceOperator([[detector_op, 0],
                                      [-odl.IdentityOperator(detector_op.domain), proj_op]])

    newrhs = bigop.range.element([projections, bigop.range[1].zero()])
    x = bigop.domain.zero()
    for i in range(20):
        cgn(bigop, x, newrhs, niter=3, partial=partial)
else:
    partial_proj = (odl.solvers.util.ShowPartial(indices=np.s_[:, :, :, n//2], clim=[0, 5]) &
                    odl.solvers.util.PrintIterationPartial('projection'))
    partial_vol = (odl.solvers.util.ShowPartial(indices=np.s_[:, :, :, n//2], clim=[0, 1]) &
                   odl.solvers.util.ShowPartial(indices=np.s_[:, n//2, :, n//2]) &
                   odl.solvers.util.PrintIterationPartial('volume'))
    partial_proj = None
    separate_projections = projections.copy() / 2.0
    volumes = proj_op.domain.zero()
    for i in range(1):
        cgn(detector_op, separate_projections, projections, niter=10,
            partial=partial_proj)
       # cgn(proj_op, volumes, separate_projections, niter=3,
       #     partial=partial_vol)
