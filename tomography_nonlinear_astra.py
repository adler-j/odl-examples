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


def make_projector(n):
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
    projector = odl.tomo.XrayTransform(discr_reco_space, geom,
                                       backend='astra_cuda')
    phantom = projector.domain.one()
    projector._adjoint *= projector(phantom).inner(projector(phantom)) / phantom.inner(projector.adjoint(projector(phantom)))
    return 0.08 * projector


def mu(E):
    return E**-4


class BeamHardeningProjector(odl.Operator):
    def __init__(self, projector, energies, spectrum):
        self.projector = projector
        self.energies = energies

        self.spectrum = spectrum
        assert np.sum(spectrum) == 1.0

        super().__init__(projector.domain, projector.range, False)

    def _call(self, x, **kwargs):
        Ahatx = kwargs.pop('Ahatx', None)
        if Ahatx is None:
            Ahatx = self.projector(x)

        result = self.range.zero()
        for E, I in zip(self.energies, self.spectrum):
            result += I * (-mu(E) * Ahatx).ufunc.exp()
        return -result.ufunc.log()

    def derivative(self, x):
        Ahatx = self.projector(x)
        Ax = self(x, Ahatx=Ahatx)

        scale = self.range.zero()
        for E, I in zip(self.energies, self.spectrum):
            scale += I * mu(E) * (Ax - mu(E) * Ahatx).ufunc.exp()

        return scale * self.projector

n = 400

# Create projector
proj = make_projector(n)

# Create nonlinear projector
energies = np.linspace(0.7, 1, 3)
spectrum = np.array([0.5, 1.0, 0.5]) / 2.0
A = BeamHardeningProjector(proj, energies, spectrum)

# Create phantom
phantom = odl.util.shepp_logan(A.domain, False)

# Create data
rhs = A(phantom)


dA = A.derivative(phantom)
norm2 = dA.adjoint(dA(phantom)).norm() / phantom.norm()

x = A.domain.one() * (phantom.ufunc.sum() / n**3)


def make_positive(x):
    x.ufunc.maximum(0.0, out=x)

partial = (odl.solvers.util.ShowPartial(clim=[0.7, 1.3]) &
           odl.solvers.util.ShowPartial(indices=np.s_[:, n//2, n//2]) &
           odl.solvers.util.PrintTimingPartial() &
           odl.solvers.util.PrintIterationPartial())

#raise Exception
odl.solvers.landweber(A, x, rhs, niter=50, omega=1.0 / norm2, partial=partial,
                      projection=make_positive)
