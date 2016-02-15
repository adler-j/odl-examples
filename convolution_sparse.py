# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import scipy
import scipy.ndimage
import odl

# Helper
import convolution_helper


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


def ind_fun(points):
    x, y = points
    return ((x-0.1)**2 + y**2 <= 0.5**2).astype(float)


def kernel(x):
    mean = [0.0, 0.0]
    std = [0.05, 0.05]
    return np.exp(-(((x[0] - mean[0]) / std[0]) ** 2 + ((x[1] - mean[1]) /
                                                        std[1]) ** 2))


def adjkernel(x):
    return kernel((-x[0], -x[1]))


# Continuous definition of problem
domain = odl.FunctionSpace(odl.Rectangle([-1, -1], [1, 1]))
kernel_domain = odl.FunctionSpace(odl.Rectangle([-2, -2], [2, 2]))

# Discretization parameters
n = 100
npoints = np.array([n+1, n+1])
npoints_kernel = np.array([2*n+1, 2*n+1])

# Discretization spaces
disc_domain = odl.uniform_discr_fromspace(domain, npoints)
disc_kernel_domain = odl.uniform_discr_fromspace(kernel_domain,
                                                 npoints_kernel)

# Discretize the functions
disc_kernel = disc_kernel_domain.element(kernel)
disc_adjkernel = disc_kernel_domain.element(adjkernel)
phantom = np.zeros([n+1, n+1])
phantom[np.random.randint(0, n+1, 10), np.random.randint(0, n+1, 10)] = 1
disc_phantom = disc_domain.element(phantom)

# Show the phantom and kernel
disc_phantom.show(title='Discrete phantom')
disc_kernel.show(title='Discrete kernel')

# Create operator (Convolution = real-space, slow; FFTConvolution = faster)
conv = convolution_helper.FFTConvolution(disc_domain, disc_kernel, disc_adjkernel)

# Sparsifying operator
Phi = odl.IdentityOperator(disc_domain)

# Verify that the operator is correctly written.

# Calculate data
data = conv(disc_phantom)

# Add noise (we can directly add a NumPy array to a discrete vector)
noisy_data = data + np.random.randn(*npoints) * 0.2 * data.ufunc.max()

# Show the data
data.show(title='Data')
noisy_data.show(title='Noisy data')

# Split bregman parameters
mu = 2e6
la = 0.5e-6 * mu

# Test split bregman
partial = odl.solvers.ShowPartial(title='solution', display_step=5)
x = disc_domain.zero()

SplitBregmanReconstruct(conv, Phi, x, noisy_data, la, mu, 5000, 1,
                        isotropic=False, partial=partial)