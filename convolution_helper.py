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
import odl

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
