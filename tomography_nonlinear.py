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
from tomography_helper import ForwardProjector


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
        for E in self.energies:
            result += (-mu(E) * Ahatx).ufunc.exp()
        return -(result / len(self.energies)).ufunc.log()

    def derivative(self, x):
        Ahatx = self.projector(x)

        Ax = self(x, Ahatx=Ahatx)

        scale = self.range.zero()
        for E in self.energies:
            scale += mu(E) * (Ax - mu(E) * Ahatx).ufunc.exp()

        return (scale / len(self.energies)) * self.projector

n = 200

# Create spaces
d = odl.uniform_discr([0, 0], [1, 1], [n, n])
ran = odl.uniform_discr([0, 0], [1, np.pi], [np.ceil(np.sqrt(2) * n), n])

# Create phantom
phantom = odl.util.shepp_logan(d, False)

# Create projector
proj = ForwardProjector(d, ran)
energies = np.linspace(0.7, 1, 3)
spectrum = np.array([0.5, 1.0, 0.5]) / 2.0
A = BeamHardeningProjector(proj, energies, spectrum)

# Create data
rhs = A(phantom)

partial = (odl.solvers.util.ShowPartial(clim=[0.7, 1.3]) &
           odl.solvers.util.ShowPartial(indices=np.s_[:, n//2]) &
           odl.solvers.util.PrintIterationPartial())

dA = A.derivative(phantom)
norm2 = dA.adjoint(dA(phantom)).norm() / phantom.norm()

x = d.one() * (phantom.ufunc.sum() / n**2)
dA = A.derivative(x)

def make_positive(x):
    x.ufunc.maximum(0.0, out=x)

odl.solvers.landweber(A, x, rhs, 100, 1.0 / norm2, partial=partial,
                      projection=make_positive)
