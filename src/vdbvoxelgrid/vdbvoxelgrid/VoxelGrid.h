#pragma once

#include <openvdb/openvdb.h>
#include <vector>

namespace vdbvoxelgrid {
using VoxelDataType = uint16_t;
using VoxelTreeType = openvdb::tree::Tree4<VoxelDataType, 5, 4, 3>::Type;
using VoxelGridType = openvdb::Grid<VoxelTreeType>;

class VoxelGrid {

public:
    struct MeshData {
        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
    };

    VoxelGrid(double voxel_size);
    ~VoxelGrid() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void Add(const std::vector<openvdb::Vec3d>& points);

    /// @brief Restore voxels directly from an (x, y, z, counts) table, e.g. one
    /// previously produced by Extract(). Sets each voxel's count at the index
    /// nearest to the given world-space center, so a grid can be reloaded
    /// without re-integrating the (much larger) original point cloud.
    void AddVoxels(const std::vector<openvdb::Vec3d>& centers,
                   const std::vector<float>& counts);

    /// @brief Occlusion query: trace a ray from @p origin toward each target in
    /// @p points and return, per target, the world-space range to the first
    /// active voxel with count >= @p min_count. The ray is capped at the target
    /// distance, so the result is always <= ||point - origin||; it equals that
    /// distance when nothing occludes the target (fail-open on an empty grid).
    /// Output is aligned 1:1 with @p points (no filtering).
    std::vector<double> RayTraceToPoints(
        const openvdb::Vec3d& origin,
        const std::vector<openvdb::Vec3d>& points,
        VoxelDataType min_count);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume. Not used by python, but useful for C++ projects.
    std::vector<double> RayTraceDepth(
        const openvdb::Mat4d& T,
        const openvdb::Mat3d& K,
        int height,
        int width,
        float max_distance,
        VoxelDataType min_counts,
        std::vector<bool> mask);

    std::vector<openvdb::Vec3d>  RayTracePoints(
        const openvdb::Mat4d& T,
        const openvdb::Mat3d& K,
        int height,
        int width,
        float max_distance,
        VoxelDataType min_counts,
        std::vector<bool> mask);

    int Length();

    /// @brief Extract active voxels with count >= @p min_count into an
    /// (x, y, z, counts) table. With the default min_count = 0 no voxels are
    /// filtered, preserving the original behavior.
    std::map<std::string, std::vector<float>> Extract(VoxelDataType min_count = 0);

    /// @brief Convert voxels with count >= @p min_count into a triangle mesh.
    /// The result contains world-space vertices and triangle indices.
    MeshData ToMesh(VoxelDataType min_count);

public:
    /// OpenVDB Grids modeling the signed distance, weight and color
    VoxelGridType::Ptr vg_;

    /// VoxelGrid public properties
    double voxel_size_;
};

}  // namespace vdbvoxelgrid
