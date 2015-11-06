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

"""Helper functions for the convolution example."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import scipy.signal
import numpy as np
import odl


class Difference(odl.Operator):
    """A operator that returns the forward difference
    """
    def __init__(self, space):
        super().__init__(space, odl.ProductSpace(space, 2), linear=True)

    def _apply(self, rhs, out):
        asarr = rhs.asarray()
        dx = asarr.copy()
        dy = asarr.copy()
        dx[: -1, :] = asarr[1:, :] - asarr[:-1, :]
        dy[:, :-1] = asarr[:, 1:] - asarr[:, :-1]

        dx[-1, :] = -asarr[-1, :]
        dy[:, -1] = -asarr[:, -1]

        out[0][:] = dx
        out[1][:] = dy

    @property
    def adjoint(self):
        return DifferenceAdjoint(self.domain)


class DifferenceAdjoint(odl.Operator):
    """A operator that returns the adjoint of the forward difference,
    the negative backwards difference
    """
    def __init__(self, space):
        super().__init__(odl.ProductSpace(space, 2), space, linear=True)

    def _apply(self, rhs, out):
        dx = rhs[0].asarray()
        dy = rhs[1].asarray()

        adj = np.zeros_like(dx)
        adj[1:, 1:] = (dx[1:, 1:] - dx[:-1, 1:]) + (dy[1:, 1:] - dy[1:, :-1])
        adj[0, 1:] = (dx[0, 1:]) + (dy[0, 1:] - dy[0, :-1])
        adj[1:, 0] = (dx[1:, 0] - dx[:-1, 0]) + (dy[1:, 0])
        adj[0, 0] = (dx[0, 0]) + (dy[0, 0])

        out[:] = -adj

    @property
    def adjoint(self):
        return Difference(self.range)


class FFTConvolution(odl.Operator):
    """ Optimized version of the convolution operator
    """
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        self.scale = kernel.space.domain.volume / len(kernel)

        super().__init__(space, space, linear=True)

    def _apply(self, rhs, out):
        out[:] = scipy.signal.fftconvolve(rhs,
                                          self.kernel,
                                          mode='same')

        out *= self.scale

    @property
    def adjoint(self):
        return FFTConvolution(self.domain, self.adjkernel, self.kernel)
