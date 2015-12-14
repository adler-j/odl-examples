""" Utility functions for tomography, simple projectors """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
from skimage.transform import radon, iradon
import numpy as np
import odl


class ForwardProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = ran.grid.meshgrid()[1][0] * 180 / np.pi
        super().__init__(dom, ran, True)

    def _call(self, x):
        scale = (self.range.grid.stride[0] *
                 self.domain.shape[0] /
                 (self.domain.shape[0] - 1.0))
        return self.range.element(radon(x.asarray(), self.theta)) * scale

    @property
    def adjoint(self):
        return BackProjector(self.range, self.domain)


class BackProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = dom.grid.meshgrid()[1][0] * 180 / np.pi
        self.npoint = ran.grid.shape[0]
        super().__init__(dom, ran, True)

    def _call(self, x):
        return self.range.element(iradon(x.asarray(), self.theta, self.npoint,
                                         filter=None))

    @property
    def adjoint(self):
        return ForwardProjector(self.range, self.domain)