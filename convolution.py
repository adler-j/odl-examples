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

class Convolution(odl.LinearOperator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        self.scale = kernel.space.domain.volume / len(kernel)
        
        super().__init__(space, space)

    def _apply(self, rhs, out):
        scipy.ndimage.convolve(rhs, 
                               self.kernel,
                               output=out,
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
data = domain.element(ind_fun)

# Discretization parameters
n = 50
nPoints = np.array([n+1, n+1])
nPointsKernel = np.array([2*n+1, 2*n+1])

# Discretization spaces
disc_domain = odl.l2_uniform_discretization(domain, nPoints)
disc_kernel_domain = odl.l2_uniform_discretization(kernel_domain, nPointsKernel)

# Discretize the functions
disc_kernel = disc_kernel_domain.element(kernel)
disc_adjkernel = disc_kernel_domain.element(adjkernel)
disc_data = disc_domain.element(data)

# Show the data and kernel
disc_data.show(title='disc_data')
disc_kernel.show(title='disc_kernel')


# Create operator
conv = Convolution(disc_domain, disc_kernel, disc_adjkernel)
#conv = FFTConvolution(disc_domain, disc_kernel, disc_adjkernel) #sped up version

# Verify that the operator is correctly written.

# Calculate result
result = conv(disc_data)

# Add noise
noisy_result = result + disc_domain.element(np.random.randn(*nPoints) * 1.0 * result.asarray().mean())

# Show the result
result.show(title='result')
noisy_result.show(title='noisy_result')

# Number of iterations
iterations = 5

# Display partial
def show_line(result):
    plt.plot(result.asarray()[:,n//2])
partial = odl.operator.solvers.ForEachPartial(show_line)

# Norm calculator used in landweber
def calc_norm(operator):
    return operator(disc_data).norm() / disc_data.norm()

# Test Landweber
plt.figure()
show_line(disc_data)

x = disc_domain.zero()
odl.operator.solvers.landweber(conv, x, noisy_result, iterations, 0.5/calc_norm(conv)**2, partial)
x.show(title='landweber')

# Test CGN
plt.figure()
show_line(disc_data)

x = disc_domain.zero()
odl.operator.solvers.conjugate_gradient_normal(conv, x, noisy_result, iterations, partial)
x.show(title='CGN')

#Tichonov reglarized conjugate gradient
Q = Difference(disc_domain)
la = 0.0001
regularized_conv = conv.T * conv + la * Q.T * Q

plt.figure()
show_line(disc_data)

x = disc_domain.zero()
odl.operator.solvers.conjugate_gradient(regularized_conv, x, conv.T(noisy_result), iterations, partial)
x.show(title='regularized')

odl.test.OpeartorTest(conv).run_tests()
odl.test.OpeartorTest(Q).run_tests()
odl.test.OpeartorTest(regularized_conv).run_tests()

plt.show()
