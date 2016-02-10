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
from odl.solvers import chambolle_pock_solver, f_cc_prox_l2_tv, g_prox_none
import matplotlib.pyplot as plt

n = 200

# Discrete reconstruction space
discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                     [n]*3, dtype='float32')

# Geometry
src_rad = 1000
det_rad = 100
angle_intvl = odl.Interval(0, 2 * np.pi)
dparams = odl.Rectangle([-50, -50], [50, 50])
agrid = odl.uniform_sampling(angle_intvl, n * 2)
dgrid = odl.uniform_sampling(dparams, [n, n])
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

# Create chambolle pock operator
grad = odl.Gradient(discr_reco_space, method='forward')
prod_op = odl.ProductSpaceOperator([[A], [grad]])

# Get norm
repeat_phatom = prod_op.domain.element([phantom, phantom])
prod_op_norm = odl.operator.oputils.power_method_opnorm(prod_op, 10,
                                                        repeat_phatom) * 2.0

# Reconstruct
partial = (odl.solvers.ShowPartial(clim=[-0.1, 1.1], display_step=1) &
           odl.solvers.ShowPartial(indices=np.s_[0, :, n//2, n//2]) &
           odl.solvers.PrintTimePartial() &
           odl.solvers.PrintIterationPartial())

# Run algorithms
rec = chambolle_pock_solver(prod_op,
                            f_cc_prox_l2_tv(prod_op.range, rhs, lam=0.01),
                            g_prox_none(prod_op.domain),
                            sigma=1 / prod_op_norm,
                            tau=1 / prod_op_norm,
                            niter=100,
                            partial=partial)[0]

# Display images
phantom.show(title='original image')
rhs.show(title='sinogram')
rec.show(title='reconstructed image')
plt.show()
