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

# Adjoint currently bugged, needs to be fixed
A._adjoint *= A(phantom).inner(A(phantom)) / phantom.inner(A.adjoint(A(phantom)))

# Create data
rhs = A(phantom)

# Add noise
mean = rhs.ufunc.sum() / rhs.size
rhs.ufunc.add(np.random.rand(A.range.size)*1.0*mean, out=rhs)

# Reconstruct
partial = (odl.solvers.util.ShowPartial(clim=[0, 1.1]) &
           odl.solvers.util.PrintIterationPartial())

x = discr_reco_space.zero()
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=100, partial=partial)
phantom.show()
