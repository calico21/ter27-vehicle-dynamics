#include "cone_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <pcl/segmentation/extract_clusters.h>

#include <patchwork/patchworkpp.h>

namespace mirena
{
    namespace
    {
        patchwork::Params to_patchwork_params(const ConeDetectorParams &p, bool has_intensity)
        {
            patchwork::Params pw;
            pw.sensor_height = p.sensor_height;
            pw.min_range = p.ground_min_range;
            pw.max_range = p.ground_max_range;
            pw.enable_RNR = p.remove_reflected_noise && has_intensity;
            pw.RNR_intensity_thr = p.rnr_intensity_thr;

            pw.num_min_pts = p.ground_num_min_pts;
            pw.th_dist = p.ground_thickness;
            pw.th_seeds = p.ground_seed_margin;
            pw.num_lpr = p.ground_num_lpr;
            pw.num_iter = p.ground_num_iter;
            pw.uprightness_thr = p.ground_uprightness_thr;
            pw.enable_RVPF = p.ground_vertical_plane_fit;
            pw.enable_TGR = p.ground_temporal_revert;

            // Only when they are the four Patchwork++ reads; the node has already
            // said so on the log if they are not.
            if (p.ground_sectors_each_zone.size() == 4 && p.ground_rings_each_zone.size() == 4)
            {
                pw.num_sectors_each_zone = p.ground_sectors_each_zone;
                pw.num_rings_each_zone = p.ground_rings_each_zone;
            }
            return pw;
        }

        double elapsed_ms(std::chrono::steady_clock::time_point since)
        {
            const std::chrono::duration<double, std::milli> d = std::chrono::steady_clock::now() - since;
            return d.count();
        }
    } // namespace

    /// Height of the track surface as a function of where you are on it, taken
    /// from the points Patchwork++ called ground.
    ///
    /// A single plane at -sensor_height is only the track if the sensor's real
    /// tilt matches its extrinsic to a fraction of a degree and the surface is
    /// level. Neither holds: a degree of unmodelled tilt puts the plane 17 cm out
    /// at 10 m, which is most of a cone, and every height test here is a
    /// centimetre-scale statement. Reading the surface back out of the ground
    /// points instead costs one pass over them and makes those tests local.
    struct ConeDetector::GroundMap
    {
        float cell = 1.0f;
        float origin_x = 0.0f;
        float origin_y = 0.0f;
        int nx = 0;
        int ny = 0;
        float fallback = 0.0f;
        std::vector<float> sum;
        std::vector<int> count;
        // The planes Patchwork++ fitted, one per patch it accepted as ground:
        // centre and normal. Used where the cells have nothing to say.
        Eigen::MatrixX3f centers;
        Eigen::MatrixX3f normals;

        float max_deviation = 1.0f;

        void build(const Eigen::MatrixX3f &ground, float cell_size, float flat_z, float deviation)
        {
            cell = cell_size > 0.0f ? cell_size : 1.0f;
            fallback = flat_z;
            max_deviation = deviation;
            sum.clear();
            count.clear();
            nx = ny = 0;

            if (ground.rows() == 0)
            {
                return;
            }

            const Eigen::Vector3f lower = ground.colwise().minCoeff();
            const Eigen::Vector3f upper = ground.colwise().maxCoeff();
            origin_x = lower.x();
            origin_y = lower.y();
            nx = static_cast<int>((upper.x() - lower.x()) / cell) + 1;
            ny = static_cast<int>((upper.y() - lower.y()) / cell) + 1;

            sum.assign(static_cast<size_t>(nx) * ny, 0.0f);
            count.assign(static_cast<size_t>(nx) * ny, 0);

            for (Eigen::Index i = 0; i < ground.rows(); ++i)
            {
                // A ground return that far off the prior is a reflection from
                // under the track, not the track.
                if (std::abs(ground(i, 2) - flat_z) > max_deviation)
                {
                    continue;
                }
                const size_t k = index_of(ground(i, 0), ground(i, 1));
                sum[k] += ground(i, 2);
                ++count[k];
            }
        }

        size_t index_of(float x, float y) const
        {
            const int ix = std::clamp(static_cast<int>((x - origin_x) / cell), 0, nx - 1);
            const int iy = std::clamp(static_cast<int>((y - origin_y) / cell), 0, ny - 1);
            return static_cast<size_t>(ix) * ny + iy;
        }

        /// The surface under (x, y): the mean of the ground returns in that cell,
        /// or of the nearest cell that has any, or the flat plane if the sweep
        /// found no ground at all.
        float at(float x, float y) const
        {
            if (nx == 0)
            {
                return fallback;
            }

            const int ix = std::clamp(static_cast<int>((x - origin_x) / cell), 0, nx - 1);
            const int iy = std::clamp(static_cast<int>((y - origin_y) / cell), 0, ny - 1);

            // Widening rings rather than one fixed neighbourhood: a cone's own
            // cell is usually filled, but a cell in the shadow of a barrier can be
            // empty for several metres and still sits on a knowable surface.
            for (int radius = 0; radius <= 3; ++radius)
            {
                float total = 0.0f;
                int n = 0;
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    for (int dy = -radius; dy <= radius; ++dy)
                    {
                        // Only the ring itself; the inside was covered already.
                        if (radius > 0 && std::abs(dx) != radius && std::abs(dy) != radius)
                        {
                            continue;
                        }
                        const int jx = ix + dx;
                        const int jy = iy + dy;
                        if (jx < 0 || jx >= nx || jy < 0 || jy >= ny)
                        {
                            continue;
                        }
                        const size_t k = static_cast<size_t>(jx) * ny + jy;
                        if (count[k] > 0)
                        {
                            total += sum[k] / static_cast<float>(count[k]);
                            ++n;
                        }
                    }
                }
                if (n > 0)
                {
                    return total / static_cast<float>(n);
                }
            }

            return from_patch_plane(x, y);
        }

        /// Where no ground was seen -- the shadow behind a barrier, a stretch the
        /// vertical field of view skips over -- extrapolate along the nearest
        /// patch plane instead of giving up and calling the track flat. The plane
        /// carries the local slope, which is the whole point of asking Patchwork++
        /// rather than assuming.
        float from_patch_plane(float x, float y) const
        {
            float best = fallback;
            float best_distance = std::numeric_limits<float>::max();

            for (Eigen::Index i = 0; i < centers.rows(); ++i)
            {
                const float dx = x - centers(i, 0);
                const float dy = y - centers(i, 1);
                const float distance = dx * dx + dy * dy;
                if (distance >= best_distance)
                {
                    continue;
                }
                // A patch that is nearly vertical cannot say where the floor is.
                const float nz = normals(i, 2);
                if (std::abs(nz) < 0.5f)
                {
                    continue;
                }
                const float height = centers(i, 2) - (normals(i, 0) * dx + normals(i, 1) * dy) / nz;
                if (std::abs(height - fallback) > max_deviation)
                {
                    continue;
                }
                best_distance = distance;
                best = height;
            }

            return best;
        }
    };

    /// Running centroid and axis-aligned bounds of one cluster, so accumulating a
    /// cluster is a single sweep over the points the extraction assigned to it.
    struct ConeDetector::ClusterBounds
    {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        Eigen::Vector3f lower = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f upper = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        int count = 0;

        void add(const Eigen::Vector3f &p)
        {
            sum += p;
            lower = lower.cwiseMin(p);
            upper = upper.cwiseMax(p);
            ++count;
        }

        Eigen::Vector3f centroid() const { return sum / static_cast<float>(count); }
        Eigen::Vector3f extent() const { return upper - lower; }
    };

    ConeDetector::ConeDetector(const ConeDetectorParams &params)
        : params_(params),
          ground_(new GroundMap),
          nonground_(new PointCloud),
          downsampled_(new PointCloud),
          // sorted=false: the extraction walks every neighbour a radius search
          // returns and cluster membership does not depend on the order they come
          // back in, so sorting each result set is pure cost -- and on a cloud
          // that still holds a wall it is most of the clustering time.
          search_(new pcl::search::KdTree<pcl::PointXYZ>(false))
    {
    }

    ConeDetector::~ConeDetector() = default;

    const std::vector<Detection> &ConeDetector::detect(const Eigen::MatrixXf &cloud_lidar,
                                                       const Eigen::Isometry3f &car_from_lidar)
    {
        const auto started = std::chrono::steady_clock::now();

        sensor_xy_ = car_from_lidar.translation().head<2>();
        detections_.clear();
        labels_.clear();
        nonground_->clear();

        if (cloud_lidar.rows() == 0)
        {
            total_ms_ = elapsed_ms(started);
            return detections_;
        }

        if (!patchwork_)
        {
            patchwork_ = std::make_unique<patchwork::PatchWorkpp>(
                to_patchwork_params(params_, cloud_lidar.cols() >= 4));
        }

        // 1. Level the cloud before ground removal: rotate by the car-from-lidar
        //    rotation but not its translation, which leaves the points around the
        //    sensor with z vertical.
        //
        //    Patchwork++ is written for a sensor whose z is up. It cuts the cloud
        //    into concentric rings and gates each fitted patch on how far its
        //    elevation sits from -sensor_height, with a ground thickness of a few
        //    centimetres. A mount tilted by t degrees tilts the whole ground plane
        //    with it, so in the raw sensor frame the ground's z climbs by
        //    range * tan(t) -- metres, over the ring widths this uses. The far
        //    patches then fail the elevation gate and the track surface is handed
        //    on as non-ground. Levelling first is what makes sensor_height mean
        //    what Patchwork++ reads it as.
        level_.resize(cloud_lidar.rows(), cloud_lidar.cols());
        level_.leftCols<3>().noalias() = cloud_lidar.leftCols<3>() * car_from_lidar.linear().transpose();
        if (cloud_lidar.cols() >= 4)
        {
            level_.rightCols(cloud_lidar.cols() - 3) = cloud_lidar.rightCols(cloud_lidar.cols() - 3);
        }

        stage_.level_ms = elapsed_ms(started);
        auto mark = std::chrono::steady_clock::now();

        patchwork_->estimateGround(level_);
        stage_.patchwork_ms = elapsed_ms(mark);
        mark = std::chrono::steady_clock::now();

        const Eigen::MatrixX3f nonground_level = patchwork_->getNonground();

        // The track surface, read back out of the points that were called ground.
        // Built in the car frame, which is where every height test below lives.
        Eigen::MatrixX3f ground_car = patchwork_->getGround();
        ground_car.rowwise() += car_from_lidar.translation().transpose();
        ground_->build(ground_car, static_cast<float>(params_.ground_cell_size),
                       car_from_lidar.translation().z() - static_cast<float>(params_.sensor_height),
                       static_cast<float>(params_.ground_max_deviation));

        // The patch planes come along for the ride, into the same frame. Only the
        // centres move: a normal is a direction, and the cloud was levelled before
        // Patchwork++ saw it, so the rotation is already in both.
        ground_->centers = patchwork_->getCenters();
        ground_->centers.rowwise() += car_from_lidar.translation().transpose();
        ground_->normals = patchwork_->getNormals();
        stage_.ground_map_ms = elapsed_ms(mark);
        mark = std::chrono::steady_clock::now();

        ground_removal_ms_ = elapsed_ms(started);

        // 2. Into the car frame -- the rotation is already in, so only the offset
        //    from the car's origin to the sensor is left -- keeping what is at a
        //    height a cone could be at and is not the car itself.
        //
        //    This is where what ground removal got wrong gets caught. Below-floor
        //    multipath comes out under the surface, the ground patches Patchwork++
        //    leaves behind sit on it, and both are gone by the floor test.
        const Eigen::Vector3f offset = car_from_lidar.translation();
        const float ego_x = static_cast<float>(params_.ego_half_length);
        const float ego_y = static_cast<float>(params_.ego_half_width);
        const float min_elevation = static_cast<float>(params_.min_elevation);
        const float max_elevation = static_cast<float>(params_.max_elevation);

        nonground_->reserve(static_cast<size_t>(nonground_level.rows()));
        for (Eigen::Index i = 0; i < nonground_level.rows(); ++i)
        {
            const float x = nonground_level(i, 0) + offset.x();
            const float y = nonground_level(i, 1) + offset.y();
            if (std::abs(x) <= ego_x && std::abs(y) <= ego_y)
            {
                continue;
            }

            const float z = nonground_level(i, 2) + offset.z();
            const float elevation = z - ground_->at(x, y);
            if (elevation < min_elevation || elevation > max_elevation)
            {
                continue;
            }
            nonground_->emplace_back(x, y, z);
        }
        nonground_->width = nonground_->size();
        nonground_->height = 1;
        // Patchwork++ never hands back a non-finite point, and the KdTree build
        // skips a pass over the cloud when it is told so.
        nonground_->is_dense = true;

        // 3. Thin the near field out. A cone at 3 m arrives as hundreds of
        //    returns and is no better localised for it, while every one of them
        //    costs a radius search whose result set is every other return on the
        //    same cone.
        returns_ = nonground_.get();
        if (params_.voxel_leaf > 0.0 && !nonground_->empty())
        {
            const float leaf = static_cast<float>(params_.voxel_leaf);
            voxel_.setInputCloud(nonground_);
            voxel_.setLeafSize(leaf, leaf, leaf);
            voxel_.filter(*downsampled_);
            // After the swap the undownsampled cloud is the one in downsampled_,
            // which is where the return counts have to be read from.
            nonground_.swap(downsampled_);
            returns_ = downsampled_.get();
        }

        labels_.assign(nonground_->size(), -1);
        stage_.band_ms = elapsed_ms(mark);
        mark = std::chrono::steady_clock::now();

        if (nonground_->empty())
        {
            total_ms_ = elapsed_ms(started);
            return detections_;
        }

        // 4. Cluster the non-ground remainder. Single-link on a KdTree: two points
        //    join the same cluster when they are within cluster_tolerance, which is
        //    the grouping a cone at range needs, since it arrives as a couple of
        //    ring segments rather than a dense blob.
        search_->setInputCloud(nonground_);
        stage_.kdtree_ms = elapsed_ms(mark);
        mark = std::chrono::steady_clock::now();

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> extraction;
        extraction.setClusterTolerance(params_.cluster_tolerance);
        // The size band is enforced here rather than by the filter below, so a wall
        // or a spectator is never accumulated only to be thrown away afterwards.
        extraction.setMinClusterSize(static_cast<pcl::uindex_t>(std::max(1, params_.min_cluster_points)));
        extraction.setMaxClusterSize(static_cast<pcl::uindex_t>(std::max(1, params_.max_cluster_points)));
        extraction.setSearchMethod(search_);
        extraction.setInputCloud(nonground_);
        extraction.extract(cluster_indices);
        stage_.cluster_ms = elapsed_ms(mark);
        mark = std::chrono::steady_clock::now();

        // 5. Accumulate each cluster, then keep the cone-shaped ones. Accepted
        //    clusters are numbered from zero as they are found, so a label doubles
        //    as an index into the published detections.
        for (const auto &cluster : cluster_indices)
        {
            ClusterBounds bounds;
            for (const auto index : cluster.indices)
            {
                bounds.add((*nonground_)[static_cast<size_t>(index)].getVector3fMap());
            }

            if (!is_cone(bounds))
            {
                continue;
            }

            const int label = static_cast<int>(detections_.size());
            detections_.push_back({bounds.centroid(), bounds.count,
                                   height_above_ground(bounds) >= params_.big_cone_min_height,
                                   position_covariance(bounds.centroid(), bounds.count)});
            for (const auto index : cluster.indices)
            {
                labels_[static_cast<size_t>(index)] = label;
            }
        }

        stage_.accept_ms = elapsed_ms(mark);
        total_ms_ = elapsed_ms(started);
        return detections_;
    }

    /// How well a cluster's centroid locates the cone, as a 2x2 row-major
    /// covariance in the sensor frame.
    ///
    /// Built in the sensor's own polar frame, where the two errors are
    /// independent and mean different things -- one along the beam, one across
    /// it -- and then rotated into x-y. That rotation is the whole reason this
    /// returns a matrix rather than a pair of numbers: a cone off to one side is
    /// uncertain diagonally, and a diagonal ellipse is not something two
    /// variances can express.
    std::array<double, 4> ConeDetector::position_covariance(
        const Eigen::Vector3f &centroid, int num_points) const
    {
        // Measured from the sensor, not from the car. Clustering runs on a cloud
        // already placed in the car frame, so "along the beam" is the direction
        // from where the LiDAR actually sits -- a metre or two forward of the
        // origin these coordinates are written in.
        const double x = centroid.x() - sensor_xy_.x();
        const double y = centroid.y() - sensor_xy_.y();
        const double range = std::hypot(x, y);
        // Straight overhead there is no meaningful bearing; the fallback keeps
        // the maths finite and the case cannot arise for a cone anyway.
        if (range < 1e-3)
        {
            const double s = params_.range_sigma * params_.range_sigma;
            return {s, 0.0, 0.0, s};
        }

        // Averaging n points over a cluster of this spread locates the centroid
        // to about spread/sqrt(n), in every direction.
        const double centroid_sigma =
            params_.cluster_spread / std::sqrt(static_cast<double>(std::max(num_points, 1)));

        const double var_along = params_.range_sigma * params_.range_sigma +
                                 centroid_sigma * centroid_sigma;
        const double var_across = std::pow(params_.bearing_sigma * range, 2.0) +
                                  centroid_sigma * centroid_sigma;

        // Rotate diag(var_along, var_across) from the beam direction into x-y.
        const double c = x / range, s = y / range;
        const double xx = c * c * var_along + s * s * var_across;
        const double yy = s * s * var_along + c * c * var_across;
        const double xy = c * s * (var_along - var_across);
        return {xx, xy, xy, yy};
    }

    /// Is this cluster the right size, and in the right place, to be a cone?
    ///
    /// Deliberately a size-and-placement test and not a shape test: at 15 m a cone
    /// is a handful of points spread over one or two rings, which carries no shape
    /// left to match against. What makes that good enough is the track -- once the
    /// ground is gone, the things on a Formula Student course that are cone-sized,
    /// cone-tall and resting on the asphalt are cones.
    float ConeDetector::height_above_ground(const ClusterBounds &cluster) const
    {
        const Eigen::Vector3f centroid = cluster.centroid();
        // Against the surface under this cluster, not under the car: on a track
        // that rises a degree over 20 m the difference is a whole cone.
        return cluster.upper.z() - ground_->at(centroid.x(), centroid.y());
    }

    bool ConeDetector::is_cone(const ClusterBounds &cluster) const
    {
        const Eigen::Vector3f extent = cluster.extent();
        const Eigen::Vector3f centroid = cluster.centroid();
        const float ground_z = ground_->at(centroid.x(), centroid.y());
        const float top = cluster.upper.z() - ground_z;
        const float base = cluster.lower.z() - ground_z;

        if (!(extent.x() <= params_.max_width &&
              extent.y() <= params_.max_width &&
              top >= params_.min_height &&
              top <= params_.max_height &&
              std::abs(base) <= params_.max_base_offset &&
              std::abs(centroid.y()) <= params_.max_lateral_offset))
        {
            return false;
        }

        // Last, because it is the only test that walks the cloud.
        return has_plausible_return_count(cluster);
    }

    bool ConeDetector::has_plausible_return_count(const ClusterBounds &cluster) const
    {
        if (params_.expected_points_max_ratio <= 0.0 || returns_ == nullptr)
        {
            return true;
        }

        const Eigen::Vector3f centroid = cluster.centroid();
        const float range = centroid.head<2>().norm();
        if (range < 1e-3f)
        {
            return true;
        }

        const double expected = params_.expected_points_at_10m *
                                std::pow(10.0 / range, params_.expected_points_exponent);
        const auto lower = static_cast<int>(expected * params_.expected_points_min_ratio);
        const auto upper = static_cast<int>(expected * params_.expected_points_max_ratio);

        // A cluster is a box in the downsampled cloud; the returns that made it
        // are the ones inside that box, give or take a voxel at each face.
        const float slack = static_cast<float>(params_.voxel_leaf);
        const Eigen::Vector3f low = cluster.lower.array() - slack;
        const Eigen::Vector3f high = cluster.upper.array() + slack;

        int count = 0;
        for (const auto &point : returns_->points)
        {
            if (point.x >= low.x() && point.x <= high.x() &&
                point.y >= low.y() && point.y <= high.y() &&
                point.z >= low.z() && point.z <= high.z())
            {
                ++count;
                if (count > upper)
                {
                    return false;
                }
            }
        }

        return count >= lower;
    }

} // namespace mirena
