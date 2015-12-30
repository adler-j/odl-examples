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
n_voxel = [n]*3
n_pixel = [n]*2
n_angle = n

# Discrete reconstruction space
discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                     n_voxel, dtype='float32')

# Geometry
src_rad = 1000
det_rad = 100
angle_intvl = odl.Interval(0, 2 * np.pi)
dparams = odl.Rectangle([-50, -50], [50, 50])
agrid = odl.uniform_sampling(angle_intvl, n_angle)
dgrid = odl.uniform_sampling(dparams, n_pixel)
geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams,
                                         src_rad, det_rad,
                                         agrid, dgrid)

# X-ray transform
A = odl.tomo.DiscreteXrayTransform(discr_reco_space, geom,
                                   backend='astra_cuda')

# Create spaces
ran = A.range

# Create phantom
phantom = odl.util.shepp_logan(discr_reco_space)

# These are tuing parameters in the algorithm
la = 3.0 / n  # Relaxation
mu = 200.0 * n  # Data fidelity

# Create projector
diff = odl.DiscreteGradient(discr_reco_space, method='forward')

# Create data
rhs = A(phantom)

# Add noise
mean = rhs.ufunc.sum() / rhs.size
rhs.ufunc.add(np.random.rand(ran.size)*0.0*mean, out=rhs)

# Reconstruct


def plot(result, fig=None):
    print(x.ufunc.max())
    plot.fig = x.show(fig=plot.fig, show=True)
plot.fig = None

partial = odl.solvers.util.ForEachPartial(plot)

x = discr_reco_space.zero()
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=50, partial=partial)
phantom.show()
