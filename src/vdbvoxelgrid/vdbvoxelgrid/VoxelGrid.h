#pragma once

#include <openvdb/openvdb.h>
#include <vector>

namespace vdbvoxelgrid {
using VoxelDataType = uint16_t;
using VoxelTreeType = openvdb::tree::Tree4<VoxelDataType, 5, 4, 3>::Type;
using VoxelGridType = openvdb::Grid<VoxelTreeType>;

class VoxelGrid {

public:
    VoxelGrid(double voxel_size);
    ~VoxelGrid() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void Add(const std::vector<openvdb::Vec3d>& points);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume. Not used by python, but useful for C++ projects.
    // std::vector<double> RayTrace(
    //     const openvdb::Mat4d& T,
    //     const openvdb::Mat3d& K,
    //     int height,
    //     int width,
    //     float max_distance,
    //     VoxelDataType min_counts,
    //     std::vector<bool> mask);

    std::tuple<std::vector<char>, std::vector<openvdb::Vec3d>, std::vector<openvdb::Coord>> RayTrace(
        const openvdb::Mat4d& T,
        const openvdb::Mat3d& K,
        int height,
        int width,
        float max_distance,
        VoxelDataType min_counts,
        std::vector<bool> mask);

    int Length();

    std::map<std::string, std::vector<int>> Extract();

public:
    /// OpenVDB Grids modeling the signed distance, weight and color
    VoxelGridType::Ptr vg_;

    /// VoxelGrid public properties
    double voxel_size_;
};

}  // namespace vdbvoxelgrid
