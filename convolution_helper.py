""" Helper functions for the convolution example
"""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import scipy.signal
import numpy as np
import odl

class Difference(odl.LinearOperator):
    """A operator that returns the forward difference
    """
    def __init__(self, space):        
        super().__init__(domain=space, range=odl.ProductSpace(space, 2))

    def _apply(self, rhs, out):
        asarr = rhs.asarray()
        dx = asarr.copy()    
        dy = asarr.copy()
        dx[:-1,:] = asarr[1:,:]-asarr[:-1,:]
        dy[:,:-1] = asarr[:,1:]-asarr[:,:-1]
        
        dx[-1,:] = -asarr[-1,:]
        dy[:,-1] = -asarr[:,-1]
        
        out[0][:] = dx
        out[1][:] = dy

    @property
    def adjoint(self):
        return DifferenceAdjoint(self.domain)
        
class DifferenceAdjoint(odl.LinearOperator):
    """A operator that returns the adjoint of the forward difference,
    the negative backwards difference
    """
    def __init__(self, space):
        super().__init__(domain=odl.ProductSpace(space, 2), range=space)

    def _apply(self, rhs, out):
        dx = rhs[0].asarray()
        dy = rhs[1].asarray()
        
        adj = np.zeros_like(dx)
        adj[1:,1:] = (dx[1:,1:] - dx[:-1,1:]) + (dy[1:,1:] - dy[1:,:-1])
        adj[0,1:] = (dx[0,1:]) + (dy[0,1:] - dy[0,:-1])
        adj[1:,0] = (dx[1:,0] - dx[:-1,0]) + (dy[1:,0])
        adj[0,0] = (dx[0,0]) + (dy[0,0])
        
        out[:] = -adj

    @property
    def adjoint(self):
        return Difference(self.range)
        
class FFTConvolution(odl.LinearOperator):
    """ Optimized version of the convolution operator
    """
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        self.scale = kernel.space.domain.volume / len(kernel)
        
        super().__init__(space, space)

    def _apply(self, rhs, out):
        out[:] = scipy.signal.fftconvolve(rhs.asarray(), 
                                          self.kernel.asarray(),
                                          mode='same')
                         
        out *= self.scale

    @property
    def adjoint(self):
        return FFTConvolution(self.domain, self.adjkernel, self.kernel)