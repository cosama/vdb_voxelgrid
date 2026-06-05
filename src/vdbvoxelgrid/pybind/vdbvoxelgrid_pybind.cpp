#include <openvdb/openvdb.h>

// pybind11
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// std stuff
#include <cstdint>
#include <memory>
#include <vector>

#include "vdbvoxelgrid/VoxelGrid.h"

namespace py = pybind11;
using namespace py::literals;

namespace vdbvoxelgrid {

openvdb::Mat4d pyarray_to_matrix4d(const py::array_t<double>& input) {
    py::buffer_info buf_info = input.request();
    if (buf_info.ndim != 2 || buf_info.shape[0] != 4 || buf_info.shape[1] != 4) {
        throw std::runtime_error("Input array must be a 2D array with shape (4, 4)");
    }
    //double* ptr = static_cast<double*>(buf_info.ptr);
    return {static_cast<double*>(buf_info.ptr)};
};
openvdb::Mat3d pyarray_to_matrix3d(const py::array_t<double>& input) {
    py::buffer_info buf_info = input.request();
    if (buf_info.ndim != 2 || buf_info.shape[0] != 3 || buf_info.shape[1] != 3) {
        throw std::runtime_error("Input array must be a 2D array with shape (3, 3)");
    }
    //double* ptr = static_cast<double*>(buf_info.ptr);
    return {static_cast<double*>(buf_info.ptr)};
};

std::vector<openvdb::Vec3d> pyarray_to_vectors3d(py::array_t<double> &arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 3) {
        throw py::cast_error("Array is wrong shape, please use (X, 3).");
    };
    std::vector<openvdb::Vec3d> eigen_vectors(arr.shape(0));
    auto arr_unchecked = arr.mutable_unchecked<2>();
    for (auto i = 0; i < arr_unchecked.shape(0); ++i) {
        eigen_vectors[i] = {&arr_unchecked(i, 0)};
    }
    return eigen_vectors;
};

template <typename T>
py::array_t<T> array1d_from_vector(std::vector<T> &&vec) {
    const auto size = static_cast<py::ssize_t>(vec.size());
    if (vec.empty()) {
        return py::array_t<T>(size);
    }

    auto vec_uptr = std::make_unique<std::vector<T>>(std::move(vec));
    T* data_ptr = vec_uptr->data();
    py::capsule vec_owner_capsule(vec_uptr.release(), [](void* ptr) {
        std::unique_ptr<std::vector<T>> released_vec(static_cast<std::vector<T>*>(ptr));
    });

    return py::array_t<T>(size, data_ptr, vec_owner_capsule);
};

template <typename T>
py::array_t<T> array2d_from_vector(std::vector<T> &&vec, py::ssize_t rows, py::ssize_t cols) {
    if (vec.empty()) {
        return py::array_t<T>(py::array::ShapeContainer{rows, cols});
    }

    auto vec_uptr = std::make_unique<std::vector<T>>(std::move(vec));
    T* data_ptr = vec_uptr->data();
    py::capsule vec_owner_capsule(vec_uptr.release(), [](void* ptr) {
        std::unique_ptr<std::vector<T>> released_vec(static_cast<std::vector<T>*>(ptr));
    });

    return py::array_t<T>(
        {rows, cols},
        data_ptr,
        vec_owner_capsule
    );
};

py::array_t<double> array2d_from_vec3d(std::vector<openvdb::Vec3d> &&vec) {
    if (vec.empty()) {
        return py::array_t<double>(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(3)});
    }

    std::vector<double> flat;
    flat.reserve(vec.size() * 3);
    for (const auto& point : vec) {
        flat.push_back(point.x());
        flat.push_back(point.y());
        flat.push_back(point.z());
    }

    return array2d_from_vector(std::move(flat), static_cast<py::ssize_t>(vec.size()), 3);
};

std::vector<bool> mask_to_bool_vector(const py::array_t<bool>& mask, int height, int width) {
    py::buffer_info buf_info = mask.request();
    if(buf_info.ndim != 2 || buf_info.shape[0] != height || buf_info.shape[1] != width){
        throw std::runtime_error("Input mask must be a 2D array with shape (height, width)");
    }

    std::vector<bool> vec_mask(height * width, true);

    auto arr_unchecked = mask.unchecked<2>();
    for (auto y = 0; y < arr_unchecked.shape(0); ++y) {
        for (auto x = 0; x < arr_unchecked.shape(1); ++x) {
            vec_mask[y * width + x] = arr_unchecked(y, x);
        }
    }
    return vec_mask;
}

PYBIND11_MODULE(vdbvoxelgrid_pybind, m) {
    py::class_<VoxelGrid, std::shared_ptr<VoxelGrid>> voxelgrid(
        m, "VoxelGrid",
        "VoxelGrid Class");
    voxelgrid
        .def(py::init<float>(), "voxel_size"_a)
        .def("add", 
            [](VoxelGrid& self, py::array_t<double>& arr){
                self.Add(pyarray_to_vectors3d(arr));
            }, "points"_a)
        .def("ray_trace_depth",
            [](
            VoxelGrid& self,
            py::array_t<double>& T,
            py::array_t<double>& K,
            int height,
            int width,
            float max_distance,
            int min_count,
            py::array_t<bool>& mask) {
                // convert mask
                auto vec_mask = mask_to_bool_vector(mask, height, width);

                // ray trace
                auto data = self.RayTraceDepth(
                    pyarray_to_matrix4d(T), pyarray_to_matrix3d(K), height, width, max_distance, min_count, vec_mask
                );

                // convert return value
                return array2d_from_vector(std::move(data), height, width);
            },
            "T"_a, "K"_a, "height"_a, "width"_a, "max_distance"_a, "min_count"_a, "mask"_a
            )
            .def("ray_trace_points",
            [](
            VoxelGrid& self,
            py::array_t<double>& T,
            py::array_t<double>& K,
            int height,
            int width,
            float max_distance,
            int min_count,
            py::array_t<bool>& mask) {
                // convert mask
                auto vec_mask = mask_to_bool_vector(mask, height, width);

                // ray trace
                auto data = self.RayTracePoints(
                    pyarray_to_matrix4d(T), pyarray_to_matrix3d(K), height, width, max_distance, min_count, vec_mask
                );

                // convert return value
                return array2d_from_vec3d(std::move(data));
            },
            "T"_a, "K"_a, "height"_a, "width"_a, "max_distance"_a, "min_count"_a, "mask"_a
            )
            .def("ray_trace_to_points",
            [](
            VoxelGrid& self,
            py::array_t<double>& origin,
            py::array_t<double>& points,
            int min_count) {
                py::buffer_info obuf = origin.request();
                if (obuf.ndim != 1 || obuf.shape[0] != 3) {
                    throw std::runtime_error("origin must be a 1D array with shape (3,)");
                }
                openvdb::Vec3d eye(static_cast<double*>(obuf.ptr));

                auto vec_points = pyarray_to_vectors3d(points);
                auto data = self.RayTraceToPoints(eye, vec_points, min_count);

                // one range per input point (1D), aligned with `points`
                return array1d_from_vector(std::move(data));
            },
            "origin"_a, "points"_a, "min_count"_a
            )
            .def("add_voxels",
            [](
            VoxelGrid& self,
            py::array_t<double>& x,
            py::array_t<double>& y,
            py::array_t<double>& z,
            py::array_t<double>& counts) {
                auto xu = x.unchecked<1>();
                auto yu = y.unchecked<1>();
                auto zu = z.unchecked<1>();
                auto cu = counts.unchecked<1>();
                const py::ssize_t n = xu.shape(0);
                if (yu.shape(0) != n || zu.shape(0) != n || cu.shape(0) != n) {
                    throw std::runtime_error("x, y, z, counts must have equal length");
                }
                std::vector<openvdb::Vec3d> centers(n);
                std::vector<float> cnts(n);
                for (py::ssize_t i = 0; i < n; ++i) {
                    centers[i] = openvdb::Vec3d(xu(i), yu(i), zu(i));
                    cnts[i] = static_cast<float>(cu(i));
                }
                self.AddVoxels(centers, cnts);
            },
            "x"_a, "y"_a, "z"_a, "counts"_a
            )
        .def("__len__",
            [](VoxelGrid& self) {
                return self.Length();
            })
        .def("to_mesh",
            [](
            VoxelGrid& self,
            int min_count) {
                auto mesh = self.ToMesh(static_cast<VoxelDataType>(min_count));

                std::vector<float> vertices;
                vertices.reserve(mesh.points.size() * 3);
                for (const auto& point : mesh.points) {
                    vertices.push_back(point.x());
                    vertices.push_back(point.y());
                    vertices.push_back(point.z());
                }

                std::vector<int32_t> faces;
                faces.reserve(mesh.triangles.size() * 3);
                for (const auto& tri : mesh.triangles) {
                    faces.push_back(static_cast<int32_t>(tri[0]));
                    faces.push_back(static_cast<int32_t>(tri[1]));
                    faces.push_back(static_cast<int32_t>(tri[2]));
                }

                pybind11::dict ret_dict;
                ret_dict["vertices"] = array2d_from_vector(std::move(vertices), mesh.points.size(), 3);
                ret_dict["faces"] = array2d_from_vector(std::move(faces), mesh.triangles.size(), 3);
                return ret_dict;
            },
            "min_count"_a
            )
        .def("extract",
            [](VoxelGrid& self, int min_count) {
                auto map = self.Extract(static_cast<VoxelDataType>(min_count));

                // this is not necessary, pybind11 can do it, but this is 5 times faster
                pybind11::dict ret_dict;
                for (auto& item : map) {
                    ret_dict[py::str(item.first)] = array1d_from_vector(std::move(item.second));
                }
                return ret_dict;
            },
            "min_count"_a = 0
            );
}

}  // namespace vdbvoxelgrid
