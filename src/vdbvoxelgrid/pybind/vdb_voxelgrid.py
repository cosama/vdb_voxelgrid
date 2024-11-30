# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from . import vdbvoxelgrid_pybind


class VoxelGrid:
    def __init__(
        self, voxel_size: float):
        self._vg = vdbvoxelgrid_pybind.VoxelGrid(
            voxel_size=float(voxel_size),
        )
        # Passthrough all data members from the C++ API
        self.voxel_size = voxel_size

    def __repr__(self) -> str:
        return (
            f"VoxelGrid with:\n"
            f"voxel_size    = {self.voxel_size}\n"
        )

    def add(self, points) -> None:
        return self._vg.add(np.asfarray(points))

    def ray_trace(self, T, K, height, width, max_distance, min_count, mask=None) -> None:
        if mask is None:
            mask = np.full((height, width), True)
        return self._vg.ray_trace(T, K, height, width, max_distance, min_count, mask)

    def extract(self):
        return self._vg.extract()

    def __len__(self):
        return len(self._vg)