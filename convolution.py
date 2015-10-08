from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ODL
import odl
from convolution_helper import Difference


class Convolution(odl.LinearOperator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        
        super().__init__(domain=space, range=space)

    def _apply(self, rhs, out):
        ndimage.convolve(rhs.ntuple.data.reshape(rhs.shape), 
                         self.kernel.ntuple.data.reshape(self.kernel.shape),
                         output=out.ntuple.data.reshape(out.shape),
                         mode='constant')

    @property
    def adjoint(self):
        return Convolution(self.domain, self.adjkernel, self.kernel)

def ind_fun(x, y):
    b = np.broadcast(x, y)
    z = np.zeros(b.shape)
    z[x**2 + y**2 <= 0.5**2] = 1
    return z

def kernel(x, y):
    return np.exp(-(x**2 + y**2)/(2*0.05**2))
    
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

# Create operator
conv = Convolution(disc_domain, disc_kernel, disc_adjkernel)

result = conv(disc_data)

noisy_result = result + disc_domain.element(np.random.randn(*nPoints) * 0.4 * result.asarray().mean())

# Dampening parameter for landweber
iterations = 10

# Display partial
def show(result):
    plt.plot(result.asarray()[:,n//2])
partial = odl.operator.solvers.ForEachPartial(show)

def calc_norm(operator):
    return operator(disc_data).norm() / disc_data.norm()

# Test Landweber
plt.figure()
show(disc_data)
odl.operator.solvers.landweber(conv, disc_domain.zero(), noisy_result, iterations, 1.0/calc_norm(conv)**2, partial)

# Test CGN
plt.figure()
show(disc_data)
odl.operator.solvers.conjugate_gradient_normal(conv, disc_domain.zero(), noisy_result, iterations, partial)

#Tichonov reglarized conjugate gradient
Q = Difference(disc_domain)
la = 20.0
regularized_conv = conv.T * conv + la * Q.T * Q
plt.figure()
show(disc_data)
odl.operator.solvers.conjugate_gradient(regularized_conv, disc_domain.zero(), conv.T(noisy_result), iterations, partial)
#odl.operator.solvers.landweber(regularized_conv, disc_domain.zero(), conv.T(noisy_result), iterations, 1.0/calc_norm(regularized_conv)**2, partial)

plt.show()
