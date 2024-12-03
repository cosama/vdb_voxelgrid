#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <vector>
// #include <functional>
// #include <tuple>

namespace vdbvoxelgrid {
using VoxelDataType = uint16_t;
using VoxelTreeType = openvdb::tree::Tree4<VoxelDataType, 5, 4, 3>::Type;
using VoxelGridType = openvdb::Grid<VoxelTreeType>;

template<class VoxelTreeType>class Tracer {
    //using VoxelGridType = openvdb::Grid<VoxelTreeType>;

public:
    Tracer(VoxelGridType::Ptr vg){ vg_ = vg; };
    ~Tracer() = default;

    std::tuple<bool, openvdb::Vec3d, openvdb::Coord> trace_single_ray(
        openvdb::Vec3d start, openvdb::Vec3d stop, int min_count
    ) {

        // get direction and distance
        auto dir = stop - start;
        double max_distance = dir.length();
        dir.normalize();

        // define the ray and the stop_time in index space
        auto ray = openvdb::math::Ray<float>(start, dir, 0, max_distance).worldToIndex(*vg_);
        auto stop_time = ray.t1();

        // Get the "unsafe" version of the grid accessors, one per thread
        auto vg_acc = vg_->getUnsafeAccessor();

        // ray trace on the leaf levels
        openvdb::math::VolumeHDDA<VoxelTreeType, openvdb::math::Ray<float>, 2> hdda;
        auto times = hdda.march(ray, vg_acc);

        // we reached the end without a hit
        if (!times.valid()) return {false, stop, openvdb::Coord()};

        // ray trace the rest on the node level
        ray.setTimes(times.t0, stop_time);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            auto is_active = vg_acc.isValueOn(voxel);
            if (is_active) {
                auto count = vg_acc.getValue(voxel);
                if (count >= min_count) {
                    return {
                        true, vg_->transform().indexToWorld(ray.eye() + ray.dir() * dda.time()), voxel
                    };
                }
            }
        } while (dda.step());
        return {false, stop, openvdb::Coord()};
    };

    std::tuple<std::vector<char>, std::vector<openvdb::Vec3d>, std::vector<openvdb::Coord>> trace_rays(
        std::vector<openvdb::Vec3d> starts, std::vector<openvdb::Vec3d> stops, int min_count
    ) {
        if (starts.size() != stops.size()) {
            throw std::runtime_error("Both starts and stops vectors need to be the same size");
        }

        std::vector<char> hits(starts.size());
        std::vector<openvdb::Vec3d> intersects(starts.size());
        std::vector<openvdb::Coord> indices(starts.size());

        #pragma omp parallel for 
        for (unsigned int i = 0; i < starts.size(); ++i) {
            auto [h, is, ind] = trace_single_ray(starts[i], stops[i], min_count);
            hits[i] = h;
            intersects[i] = is;
            indices[i] = ind;
        };
        return {hits, intersects, indices};
    };

    std::tuple<std::vector<char>, std::vector<openvdb::Vec3d>, std::vector<openvdb::Coord>> ray_trace_image(
        const openvdb::Mat4d& T,    
        const openvdb::Mat3d& K,
        int height,
        int width,
        float max_distance,
        VoxelDataType min_count,
        std::vector<bool> mask
    ) {

        double f_x = K(0, 0);
        double f_y = K(1, 1);
        double c_x = K(0, 2);
        double c_y = K(1, 2);

        openvdb::Mat3d rot(
            T(0, 0), T(0, 1), T(0, 2),
            T(1, 0), T(1, 1), T(1, 2),
            T(2, 0), T(2, 1), T(2, 2)
        );
        openvdb::Vec3d start(T(0, 3), T(1, 3), T(2, 3));

        std::vector<openvdb::Vec3d> starts;
        std::vector<openvdb::Vec3d> stops;

        for (int y = 0; y < height; ++y) {
            double y_stop = start[1] + (y - c_y) * max_distance / f_y;
            for (int x = 0; x < width; ++x) {
                if (!mask[y * width + x]) continue;
                double x_stop = start[0] + (x - c_x) * max_distance / f_x;

                starts.push_back(start);
                stops.emplace_back(x_stop, y_stop, start[2] + max_distance);
            }
        }

        return trace_rays(starts, stops, min_count);
    };

private:
    /// OpenVDB Grids modeling the signed distance, weight and color
    VoxelGridType::Ptr vg_;

};


}  // namespace vdbvoxelgrid
