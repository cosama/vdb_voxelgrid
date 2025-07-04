#include <openvdb/openvdb.h>

// pybind11
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// std stuff
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
py::array_t<double> array2d_from_vector(std::vector<T> &&vec, py::ssize_t rows, py::ssize_t cols) {
    if (vec.empty()) {
        return py::array_t<double>();
    }

    double* data_ptr = reinterpret_cast<double*>(vec.data());

    auto vec_uptr = std::make_unique<std::vector<T>>(std::move(vec));
    py::capsule vec_owner_capsule(vec_uptr.release(), [](void* ptr) {
        std::unique_ptr<std::vector<T>> released_vec(static_cast<std::vector<T>*>(ptr));
    });

    return py::array_t<double>(
        {rows, cols},
        data_ptr,
        vec_owner_capsule
    );
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
                return array2d_from_vector(std::move(data), data.size(), 3);
            },
            "T"_a, "K"_a, "height"_a, "width"_a, "max_distance"_a, "min_count"_a, "mask"_a
            )
        .def("__len__",
            [](VoxelGrid& self) {
                return self.Length();
            })
        .def("extract",
            [](VoxelGrid& self) {
                auto map = self.Extract();

                // this is not necessary, pybind11 can do it, but this is 5 times faster
                pybind11::dict ret_dict;
                for (auto it = map.begin(); it != map.end(); it++) {
                    ret_dict[py::str(it->first)] = py::array_t<int>(it->second.size(), it->second.data());
                }
                return ret_dict;
            });
}

}  // namespace vdbvoxelgrid
