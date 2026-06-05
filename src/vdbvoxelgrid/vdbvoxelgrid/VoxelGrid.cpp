#include "VoxelGrid.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>

#include <array>
#include <unordered_map>
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


void VoxelGrid::AddVoxels(const std::vector<openvdb::Vec3d>& centers,
                          const std::vector<float>& counts) {
    if (centers.size() != counts.size()) {
        std::cerr << "AddVoxels: centers and counts must have equal length\n";
        return;
    }

    const openvdb::math::Transform& xform = vg_->transform();
    auto vg_acc = vg_->getUnsafeAccessor();

    for (size_t i = 0; i < centers.size(); ++i) {
        auto v3ijk = xform.worldToIndex(centers[i]);
        openvdb::Coord ijk(std::round(v3ijk.x()), std::round(v3ijk.y()), std::round(v3ijk.z()));
        vg_acc.setValue(ijk, static_cast<VoxelDataType>(counts[i]));
    }
}


// function to return the voxelgrid
// std::map<std::string, std::vector<int>> VoxelGrid::Extract(){
//     int length = vg_->activeVoxelCount();
//     std::vector<int> counts(length);
//     std::vector<int> ix(length);
//     std::vector<int> iy(length);
//     std::vector<int> iz(length);

//     int i = 0;
//     for (auto iter = vg_->cbeginValueOn(); iter; ++iter) {
//         float count = iter.getValue();
//         auto ijk = iter.getCoord();
//         counts[i] = count;
//         ix[i] = ijk[0];
//         iy[i] = ijk[1];
//         iz[i] = ijk[2];
//         i++;
//     }
//     return {{"counts", counts}, {"ix", ix}, {"iy", iy}, {"iz", iz}};
// }
std::map<std::string, std::vector<float>> VoxelGrid::Extract(VoxelDataType min_count)
{
    const int length = static_cast<int>(vg_->activeVoxelCount());

    std::vector<float> counts;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    counts.reserve(length);
    x.reserve(length);
    y.reserve(length);
    z.reserve(length);

    const openvdb::math::Transform& xform = vg_->transform();

    for (auto iter = vg_->cbeginValueOn(); iter; ++iter) {
        if (iter.getValue() < min_count) {
            continue;
        }

        float count = static_cast<float>(iter.getValue());
        counts.push_back(count);

        const openvdb::Coord& ijk = iter.getCoord();

        const openvdb::Vec3d worldPos = xform.indexToWorld(ijk);
        x.push_back(static_cast<float>(worldPos.x()));
        y.push_back(static_cast<float>(worldPos.y()));
        z.push_back(static_cast<float>(worldPos.z()));
    }

    return {
        {"counts", std::move(counts)},
        {"x",      std::move(x)},
        {"y",      std::move(y)},
        {"z",      std::move(z)}
    };
}


// Function to generate rays from camera parameters
std::vector<double> VoxelGrid::RayTraceDepth(
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

    std::vector<double> data(height * width, max_distance);

    #pragma omp parallel for 
    for (int y = 0; y < height; ++y) {
        double normalized_y = (y - c_y) / f_y;
        // each thread needs its own hdda
        openvdb::math::VolumeHDDA<VoxelTreeType, openvdb::math::Ray<float>, 2> hdda;
        // Get the "unsafe" version of the grid accessors, one per thread
        auto vg_acc = vg_->getUnsafeAccessor();
        for (int x = 0; x < width; ++x) {

            if(!mask[y * width + x]) continue;
            double normalized_x = (x - c_x) / f_x;

            openvdb::Vec3d dir_camera(normalized_x, normalized_y, 1.0);
            dir_camera.normalize();
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


std::vector<openvdb::Vec3d> VoxelGrid::RayTracePoints(
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

    // Store masked (x, y) coordinates
    std::vector<std::pair<int, int>> masked_pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (mask[y * width + x]) {
                masked_pixels.push_back({x, y});
            }
        }
    }

    std::vector<openvdb::Vec3d> intersection_points(masked_pixels.size());
    std::vector<bool> hit_flags(masked_pixels.size(), false); // To track if a hit occurred

    #pragma omp parallel for 
    for (unsigned int i = 0; i < masked_pixels.size(); ++i) {
        int x = masked_pixels[i].first;
        int y = masked_pixels[i].second;

        double normalized_y = (y - c_y) / f_y;
        openvdb::math::VolumeHDDA<VoxelTreeType, openvdb::math::Ray<float>, 2> hdda;
        auto vg_acc = vg_->getUnsafeAccessor();

        double normalized_x = (x - c_x) / f_x;

        openvdb::Vec3d dir_camera(normalized_x, normalized_y, 1.0);
        dir_camera.normalize();
        auto dir = rot * dir_camera;

        auto ray = openvdb::math::Ray<float>(eye, dir, 0, max_distance).worldToIndex(*vg_);
        auto end_time = ray.t1();

        auto times = hdda.march(ray, vg_acc);

        if (!times.valid()) continue;

        ray.setTimes(times.t0, end_time);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            auto is_active = vg_acc.isValueOn(voxel);
            if (is_active) {
                auto count = vg_acc.getValue(voxel);
                if (count >= min_count) {
                    intersection_points[i] = eye + dir * dda.time() * voxel_size_;
                    hit_flags[i] = true;
                    break;
                }
            }
        } while (dda.step());
    }

    // Filter out non-hits and return the valid intersection points
    std::vector<openvdb::Vec3d> final_intersection_points;
    for (unsigned int i = 0; i < masked_pixels.size(); ++i) {
        if (hit_flags[i]) {
            final_intersection_points.push_back(intersection_points[i]);
        }
    }

    return final_intersection_points;
}

std::vector<double> VoxelGrid::RayTraceToPoints(
    const openvdb::Vec3d& origin,
    const std::vector<openvdb::Vec3d>& points,
    VoxelDataType min_count) {

    std::vector<double> ranges(points.size());

    #pragma omp parallel for
    for (unsigned int i = 0; i < points.size(); ++i) {
        openvdb::Vec3d dir = points[i] - origin;
        double L = dir.length();
        // Default: target is fully visible (range == distance to target). This is
        // also the fail-open answer for a zero-length segment or an empty grid.
        ranges[i] = L;
        if (L <= 0.0) continue;
        dir /= L;

        // Cap the ray at the target distance L (dir is unit, so the parametric
        // time equals world distance, consistent with dda.time() * voxel_size_).
        auto ray = openvdb::math::Ray<float>(origin, dir, 0, L).worldToIndex(*vg_);
        auto end_time = ray.t1();

        // each thread needs its own hdda / accessor
        openvdb::math::VolumeHDDA<VoxelTreeType, openvdb::math::Ray<float>, 2> hdda;
        auto vg_acc = vg_->getUnsafeAccessor();

        auto times = hdda.march(ray, vg_acc);
        if (!times.valid()) continue;

        ray.setTimes(times.t0, end_time);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            if (vg_acc.isValueOn(voxel)) {
                auto count = vg_acc.getValue(voxel);
                if (count >= min_count) {
                    ranges[i] = dda.time() * voxel_size_;
                    break;
                }
            }
        } while (dda.step());
    }

    return ranges;
}

int VoxelGrid::Length(){
    return vg_->activeVoxelCount();
}

VoxelGrid::MeshData VoxelGrid::ToMesh(VoxelDataType min_count) {
    MeshData mesh_data;
    mesh_data.points.reserve(static_cast<size_t>(vg_->activeVoxelCount()) * 8);
    mesh_data.triangles.reserve(static_cast<size_t>(vg_->activeVoxelCount()) * 12);

    struct CornerKey {
        int x;
        int y;
        int z;

        bool operator==(const CornerKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct CornerKeyHash {
        size_t operator()(const CornerKey& key) const noexcept {
            const size_t hx = std::hash<int>{}(key.x);
            const size_t hy = std::hash<int>{}(key.y);
            const size_t hz = std::hash<int>{}(key.z);
            return hx ^ (hy << 1U) ^ (hz << 2U);
        }
    };

    std::unordered_map<CornerKey, size_t, CornerKeyHash> vertex_lookup;
    vertex_lookup.reserve(static_cast<size_t>(vg_->activeVoxelCount()) * 8);

    const auto& xform = vg_->transform();
    auto vg_acc = vg_->getConstAccessor();

    auto get_vertex = [&](int cx, int cy, int cz) -> size_t {
        CornerKey key{cx, cy, cz};
        auto it = vertex_lookup.find(key);
        if (it != vertex_lookup.end()) {
            return it->second;
        }

        const openvdb::Vec3d index_pos(
            static_cast<double>(cx) * 0.5,
            static_cast<double>(cy) * 0.5,
            static_cast<double>(cz) * 0.5);
        const openvdb::Vec3d world_pos = xform.indexToWorld(index_pos);
        const size_t index = mesh_data.points.size();
        mesh_data.points.emplace_back(
            static_cast<float>(world_pos.x()),
            static_cast<float>(world_pos.y()),
            static_cast<float>(world_pos.z()));
        vertex_lookup.emplace(key, index);
        return index;
    };

    auto emit_quad = [&](size_t v0, size_t v1, size_t v2, size_t v3) {
        mesh_data.triangles.emplace_back(
            static_cast<int>(v0), static_cast<int>(v1), static_cast<int>(v2));
        mesh_data.triangles.emplace_back(
            static_cast<int>(v0), static_cast<int>(v2), static_cast<int>(v3));
    };

    for (auto iter = vg_->cbeginValueOn(); iter; ++iter) {
        if (iter.getValue() < min_count) {
            continue;
        }

        const openvdb::Coord ijk = iter.getCoord();
        const int i = ijk.x();
        const int j = ijk.y();
        const int k = ijk.z();

        const bool xm = !vg_acc.isValueOn(openvdb::Coord(i - 1, j, k)) ||
            vg_acc.getValue(openvdb::Coord(i - 1, j, k)) < min_count;
        const bool xp = !vg_acc.isValueOn(openvdb::Coord(i + 1, j, k)) ||
            vg_acc.getValue(openvdb::Coord(i + 1, j, k)) < min_count;
        const bool ym = !vg_acc.isValueOn(openvdb::Coord(i, j - 1, k)) ||
            vg_acc.getValue(openvdb::Coord(i, j - 1, k)) < min_count;
        const bool yp = !vg_acc.isValueOn(openvdb::Coord(i, j + 1, k)) ||
            vg_acc.getValue(openvdb::Coord(i, j + 1, k)) < min_count;
        const bool zm = !vg_acc.isValueOn(openvdb::Coord(i, j, k - 1)) ||
            vg_acc.getValue(openvdb::Coord(i, j, k - 1)) < min_count;
        const bool zp = !vg_acc.isValueOn(openvdb::Coord(i, j, k + 1)) ||
            vg_acc.getValue(openvdb::Coord(i, j, k + 1)) < min_count;

        const int x0 = 2 * i - 1;
        const int x1 = 2 * i + 1;
        const int y0 = 2 * j - 1;
        const int y1 = 2 * j + 1;
        const int z0 = 2 * k - 1;
        const int z1 = 2 * k + 1;

        if (xm) {
            emit_quad(
                get_vertex(x0, y0, z0),
                get_vertex(x0, y0, z1),
                get_vertex(x0, y1, z1),
                get_vertex(x0, y1, z0));
        }
        if (xp) {
            emit_quad(
                get_vertex(x1, y0, z0),
                get_vertex(x1, y1, z0),
                get_vertex(x1, y1, z1),
                get_vertex(x1, y0, z1));
        }
        if (ym) {
            emit_quad(
                get_vertex(x0, y0, z0),
                get_vertex(x1, y0, z0),
                get_vertex(x1, y0, z1),
                get_vertex(x0, y0, z1));
        }
        if (yp) {
            emit_quad(
                get_vertex(x0, y1, z0),
                get_vertex(x0, y1, z1),
                get_vertex(x1, y1, z1),
                get_vertex(x1, y1, z0));
        }
        if (zm) {
            emit_quad(
                get_vertex(x0, y0, z0),
                get_vertex(x0, y1, z0),
                get_vertex(x1, y1, z0),
                get_vertex(x1, y0, z0));
        }
        if (zp) {
            emit_quad(
                get_vertex(x0, y0, z1),
                get_vertex(x1, y0, z1),
                get_vertex(x1, y1, z1),
                get_vertex(x0, y1, z1));
        }
    }

    return mesh_data;
}

}  // namespace vdbvoxelgrid
