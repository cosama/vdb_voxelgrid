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
    def __init__(self, voxel_size: float):
        self._vg = vdbvoxelgrid_pybind.VoxelGrid(
            voxel_size=float(voxel_size),
        )
        # Passthrough all data members from the C++ API
        self.voxel_size = voxel_size

    def __repr__(self) -> str:
        return f"VoxelGrid with:\n" f"voxel_size    = {self.voxel_size}\n"

    def add(self, points) -> None:
        return self._vg.add(np.asarray(points, dtype=float, order="C"))

    def ray_trace_depth(self, T, K, height, width, max_distance, min_count, mask=None) -> None:
        T = np.asarray(T, dtype=float, order="C")
        K = np.asarray(K, dtype=float, order="C")
        if mask is None:
            mask = np.full((height, width), True)
        return self._vg.ray_trace_depth(T, K, height, width, max_distance, min_count, mask)

    def ray_trace_points(self, T, K, height, width, max_distance, min_count, mask=None) -> None:
        T = np.asarray(T, dtype=float, order="C")
        K = np.asarray(K, dtype=float, order="C")
        if mask is None:
            mask = np.full((height, width), True)
        return self._vg.ray_trace_points(T, K, height, width, max_distance, min_count, mask)

    def add_voxels(self, x, y, z, counts) -> None:
        """Restore voxels directly from an (x, y, z, counts) world-space table,
        e.g. one produced by :meth:`extract`."""
        return self._vg.add_voxels(
            np.asarray(x, dtype=float, order="C"),
            np.asarray(y, dtype=float, order="C"),
            np.asarray(z, dtype=float, order="C"),
            np.asarray(counts, dtype=float, order="C"),
        )

    def ray_trace_to_points(self, origin, points, min_count):
        """Occlusion query: per target in ``points`` (N, 3), the range from
        ``origin`` (3,) to the first occupied voxel along the ray, capped at the
        target distance (== target distance when unoccluded)."""
        return self._vg.ray_trace_to_points(
            np.asarray(origin, dtype=float, order="C"),
            np.asarray(points, dtype=float, order="C"),
            int(min_count),
        )

    def extract(self, min_count=0):
        return self._vg.extract(int(min_count))

    def to_mesh(self, min_count):
        """Convert voxels with count >= ``min_count`` into a triangle mesh.

        Returns a dict with ``vertices`` shaped (N, 3) and ``faces`` shaped
        (M, 3), both as NumPy arrays.
        """
        mesh_fn = getattr(self._vg, "to_mesh", None)
        if callable(mesh_fn):
            return mesh_fn(int(min_count))

        # Fallback for older extension builds that do not yet expose the C++
        # mesh method. Keep the same voxel-surface extraction semantics: emit
        # only exposed cube faces for voxels whose count crosses the threshold.
        vox = self.extract()
        counts = np.asarray(vox.get("counts", ()), dtype=float)
        if counts.size == 0:
            return {
                "vertices": np.zeros((0, 3), dtype=np.float32),
                "faces": np.zeros((0, 3), dtype=np.int32),
            }

        x = np.asarray(vox["x"], dtype=float)
        y = np.asarray(vox["y"], dtype=float)
        z = np.asarray(vox["z"], dtype=float)
        keep = counts >= float(min_count)
        x, y, z = x[keep], y[keep], z[keep]
        if x.size == 0:
            return {
                "vertices": np.zeros((0, 3), dtype=np.float32),
                "faces": np.zeros((0, 3), dtype=np.int32),
            }

        voxel = float(self.voxel_size)
        centers = np.c_[x, y, z]
        center_keys = [tuple(np.round(c / voxel).astype(int)) for c in centers]
        center_by_key = {k: c for k, c in zip(center_keys, centers, strict=False)}
        center_set = set(center_by_key)

        vertex_lookup: dict[tuple[float, float, float], int] = {}
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        def get_vertex(pos: np.ndarray) -> int:
            key = tuple(np.round(pos, 9))
            idx = vertex_lookup.get(key)
            if idx is not None:
                return idx
            idx = len(vertices)
            vertices.append([float(pos[0]), float(pos[1]), float(pos[2])])
            vertex_lookup[key] = idx
            return idx

        # For each occupied voxel, emit the 6 cube faces that are not shared by
        # another occupied voxel.
        for cx, cy, cz in center_keys:
            center = center_by_key[(cx, cy, cz)]
            for dx, dy, dz, corners in (
                (-1, 0, 0, [(-1, -1, -1), (-1, -1, +1), (-1, +1, +1), (-1, +1, -1)]),
                (+1, 0, 0, [(+1, -1, -1), (+1, +1, -1), (+1, +1, +1), (+1, -1, +1)]),
                (0, -1, 0, [(-1, -1, -1), (+1, -1, -1), (+1, -1, +1), (-1, -1, +1)]),
                (0, +1, 0, [(-1, +1, -1), (-1, +1, +1), (+1, +1, +1), (+1, +1, -1)]),
                (0, 0, -1, [(-1, -1, -1), (-1, +1, -1), (+1, +1, -1), (+1, -1, -1)]),
                (0, 0, +1, [(-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1)]),
            ):
                neighbor = (cx + dx, cy + dy, cz + dz)
                if neighbor in center_set:
                    continue
                quad = [
                    get_vertex(center + voxel * 0.5 * np.asarray((ox, oy, oz), dtype=float))
                    for ox, oy, oz in corners
                ]
                faces.append([quad[0], quad[1], quad[2]])
                faces.append([quad[0], quad[2], quad[3]])

        return {
            "vertices": np.asarray(vertices, dtype=np.float32),
            "faces": np.asarray(faces, dtype=np.int32),
        }

    def __len__(self):
        return len(self._vg)
