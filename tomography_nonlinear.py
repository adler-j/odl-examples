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
    def __init__(self, projector, energies):
        self.projector = projector
        self.energies = energies
        super().__init__(projector.domain, projector.range, False)

    def _call(self, x):
        Ax = self.projector(x)

        result = self.range.zero()
        for E in self.energies:
            result += (-mu(E) * Ax).ufunc.exp()
        return -(result / len(self.energies)).ufunc.log()

    def derivative(self, x):
        Ax = self(x)
        Ahatx = self.projector(x)
        scale = self.range.zero()
        for E in self.energies:
            scale += mu(E) * (Ax - mu(E) * Ahatx).ufunc.exp()

        return (scale / len(self.energies)) * self.projector

n = 200

# Create spaces
d = odl.uniform_discr([0, 0], [1, 1], [n, n])
ran = odl.uniform_discr([0, 0], [1, np.pi], [np.ceil(np.sqrt(2) * n), n])

# Create phantom
phantom = odl.util.shepp_logan(d)

# Create projector
proj = ForwardProjector(d, ran)
A = BeamHardeningProjector(proj, np.linspace(0.3, 1, 10))

# Create data
rhs = A(phantom)


# Reconstruct
def fun(x):
    x.show()

partial = odl.solvers.util.ForEachPartial(fun)

x = d.one() * phantom.ufunc.mean()
#odl.solvers.landweber(A.derivative(x), x, rhs, 5, 0.02)#
odl.solvers.landweber(A, x, rhs, 100, 0.03, partial)
x.show()
x.show(indices=np.s_[:,50])

