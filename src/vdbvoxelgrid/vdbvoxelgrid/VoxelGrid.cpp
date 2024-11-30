#include "VoxelGrid.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>

#include <vector>

namespace vdbvoxelgrid {

VoxelGrid::VoxelGrid(double voxel_size)
    : voxel_size_(voxel_size) {

    vg_ = VoxelGridType::create(0);
    vg_->setName("V(x): Voxelgrid");
    vg_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    vg_->setGridClass(openvdb::GRID_UNKNOWN);
}

void VoxelGrid::Add(const std::vector<openvdb::Vec3d>& points) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = vg_->transform();

    // Get the "unsafe" version of the grid accessors
    auto vg_acc = vg_->getUnsafeAccessor();

    // Iterate points
    std::for_each(points.cbegin(), points.cend(), [&](const auto& xyz) {
        // openvdb::math::Vec3d xyz(point.x(), point.y(), point.z());
        auto v3ijk = xform.worldToIndex(xyz);
        openvdb::Coord ijk(std::round(v3ijk.x()), std::round(v3ijk.y()), std::round(v3ijk.z()));
        vg_acc.setValue(ijk, vg_acc.getValue(ijk) + (int32_t)1);
    });
}


// function to return the voxelgrid
std::map<std::string, std::vector<int>> VoxelGrid::Extract(){
    int length = vg_->activeVoxelCount();
    std::vector<int> counts(length);
    std::vector<int> ix(length);
    std::vector<int> iy(length);
    std::vector<int> iz(length);

    int i = 0;
    for (auto iter = vg_->cbeginValueOn(); iter; ++iter) {
        float count = iter.getValue();
        auto ijk = iter.getCoord();
        counts[i] = count;
        ix[i] = ijk[0];
        iy[i] = ijk[1];
        iz[i] = ijk[2];
        i++;
    }
    return {{"counts", counts}, {"ix", ix}, {"iy", iy}, {"iz", iz}};
}


// Function to generate rays from camera parameters
std::vector<double> VoxelGrid::RayTrace(
    const openvdb::Mat4d& T,    
    const openvdb::Mat3d& K,
    int height,
    int width,
    float max_distance,
    VoxelDataType min_count,
    std::vector<bool> mask) {

    double f_x = K(0, 0);
    double f_y = K(1, 1);
    double c_x = K(0, 2);
    double c_y = K(1, 2);

    openvdb::Mat3d rot(
        T(0, 0), T(0, 1), T(0, 2),
        T(1, 0), T(1, 1), T(1, 2),
        T(2, 0), T(2, 1), T(2, 2)
    );
    openvdb::Vec3d eye(T(0, 3), T(1, 3), T(2, 3));

    // pybind11::array_t<double> arr({width, height});
    // double* data_ptr = arr.mutable_unchecked<2>().data();
    std::vector<double> data(height * width, max_distance);

    #pragma omp parallel for 
    for (int y = 0; y < height; ++y) {
        double normalized_y = (y - c_y) / f_y;
        // Get the "unsafe" version of the grid accessors
        openvdb::math::VolumeHDDA<VoxelTreeType, openvdb::math::Ray<float>, 2> hdda;
        auto vg_acc = vg_->getUnsafeAccessor();
        for (int x = 0; x < width; ++x) {

            //data[y * width + x] = max_distance;
            if(!mask[y * width + x]) continue;
            double normalized_x = (x - c_x) / f_x;

            openvdb::Vec3d dir_camera(normalized_x, normalized_y, 1.0);
            auto dir = rot * dir_camera;

            auto ray = openvdb::math::Ray<float>(eye, dir, 0, max_distance).worldToIndex(*vg_);
            auto end_time = ray.t1();

            // ray trace on the leaf levels
            auto times = hdda.march(ray, vg_acc);

            // we reached the end without a hit
            if (!times.valid()) continue;

            // ray trace the rest on the node level
            ray.setTimes(times.t0, end_time);
            openvdb::math::DDA<decltype(ray)> dda(ray);
            do {
                const auto voxel = dda.voxel();
                auto is_active = vg_acc.isValueOn(voxel);
                if (is_active) {
                    auto count = vg_acc.getValue(voxel);
                    if (count >= min_count) {
                        data[y * width + x] = dda.time() * voxel_size_;
                        break;
                    }
                }
            } while (dda.step());
            // std::vector<openvdb::math::Ray<float>::TimeSpan> hits;
            // hdda.hits(ray, tsdf_acc, hits);
        }
    }

    return data;
}

int VoxelGrid::Length(){
    return vg_->activeVoxelCount();
}

}  // namespace vdbvoxelgrid
