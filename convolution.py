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
from builtins import super

# External
import numpy as np
import matplotlib.pyplot as plt
import scipy
import odl

# Helper
from convolution_helper import Difference, FFTConvolution


class Convolution(odl.Operator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        self.scale = kernel.space.domain.volume / len(kernel)

        super().__init__(space, space, linear=True)

    def _apply(self, rhs, out):
        scipy.ndimage.convolve(rhs,
                               self.kernel,
                               output=out.asarray(),
                               mode='constant')

        out *= self.scale

    @property
    def adjoint(self):
        return Convolution(self.domain, self.adjkernel, self.kernel)


def ind_fun(x, y):
    return ((x-0.1)**2 + y**2 <= 0.5**2).astype(float)


def kernel(x, y):
    mean = [0.0, 0.25]
    std = [0.05, 0.05]
    return np.exp(-(((x-mean[0])/std[0])**2 + ((y-mean[1])/std[1])**2))


def adjkernel(x, y):
    return kernel(-x, -y)


# Continuous definition of problem
domain = odl.L2(odl.Rectangle([-1, -1], [1, 1]))
kernel_domain = odl.L2(odl.Rectangle([-2, -2], [2, 2]))

# Complicated functions to check performance
kernel = kernel_domain.element(kernel)
adjkernel = kernel_domain.element(adjkernel)
phantom = domain.element(ind_fun)

# Discretization parameters
n = 50
npoints = np.array([n+1, n+1])
npoints_kernel = np.array([2*n+1, 2*n+1])

# Discretization spaces
disc_domain = odl.l2_uniform_discretization(domain, npoints)
disc_kernel_domain = odl.l2_uniform_discretization(kernel_domain,
                                                   npoints_kernel)

# Discretize the functions
disc_kernel = disc_kernel_domain.element(kernel)
disc_adjkernel = disc_kernel_domain.element(adjkernel)
disc_phantom = disc_domain.element(phantom)

# Show the phantom and kernel
disc_phantom.show(title='Discrete phantom')
disc_kernel.show(title='Discrete kernel')

# Create operator (Convolution = real-space, slow; FFTConvolution = faster)
conv = Convolution(disc_domain, disc_kernel, disc_adjkernel)
# conv = FFTConvolution(disc_domain, disc_kernel, disc_adjkernel)

# Verify that the operator is correctly written.

# Calculate data
data = conv(disc_phantom)

# Add noise (we can directly add a NumPy array to a discrete vector)
noisy_data = data + np.random.randn(*npoints) * data.asarray().mean()

# Show the data
data.show(title='Data')
noisy_data.show(title='Noisy data')

# Number of iterations
iterations = 5


# Display partial
def show_line(data):
    plt.plot(data.asarray()[:, n//2])
partial = odl.operator.solvers.ForEachPartial(show_line)


# Norm calculator used in landweber
def calc_norm(operator):
    return operator(disc_phantom).norm() / disc_phantom.norm()


# Test Landweber
plt.figure()
show_line(disc_phantom)

x = disc_domain.zero()
odl.operator.solvers.landweber(
    conv, x, noisy_data, niter=iterations, omega=0.5/calc_norm(conv)**2,
    partial=partial)
x.show(title='Landweber solution')

# Test CGN
plt.figure()
show_line(disc_phantom)

x = disc_domain.zero()
odl.operator.solvers.conjugate_gradient_normal(
    conv, x, noisy_data, niter=iterations, partial=partial)
x.show(title='CGN')

# Tikhonov reglarized conjugate gradient (C*C + lambda*Q*Q)
Q = Difference(disc_domain)
la = 0.0001
regularized_conv = conv.adjoint * conv + la * Q.adjoint * Q

plt.figure()
show_line(disc_phantom)

x = disc_domain.zero()
odl.operator.solvers.conjugate_gradient(
    regularized_conv, x, conv.adjoint(noisy_data),
    niter=iterations, partial=partial)
x.show(title='Regularized Landweber solution')

odl.test.OpeartorTest(conv).run_tests()
odl.test.OpeartorTest(Q).run_tests()
odl.test.OpeartorTest(regularized_conv).run_tests()

plt.show()
