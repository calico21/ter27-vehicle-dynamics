#ifndef MIRENA_PERCEPTION_CONE_DETECTOR_HPP
#define MIRENA_PERCEPTION_CONE_DETECTOR_HPP

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

// Forward declared on purpose: patchworkpp.h opens with a global
// `using namespace std;`, so it is included by cone_detector.cpp alone.
namespace patchwork
{
    class PatchWorkpp;
}

namespace mirena
{
    /// Everything the geometry pipeline needs, mirroring the node's ROS
    /// parameters one for one so the node stays a thin translation layer.
    struct ConeDetectorParams
    {
        // -- Ground removal (Patchwork++) ---------------------------------------
        /// Height of the LiDAR above the track surface. Patchwork++ measures every
        /// elevation against it, and it is what places the ground plane in the car
        /// frame, so it is the one value that has to match the real car.
        double sensor_height = 1.0;
        /// Patchwork++ *discards* points outside this radius band rather than
        /// calling them non-ground, so the band is the detector's field of view.
        /// The upstream default min_range of 2.7 m would silently swallow every
        /// cone the car is about to drive past.
        double ground_min_range = 0.5;
        double ground_max_range = 50.0;
        /// Reflected-noise removal needs an intensity channel; it is enabled only
        /// when the incoming cloud actually carries one.
        bool remove_reflected_noise = true;
        /// A point is reflected noise when it is both steeply downward -- below
        /// 15 degrees under the sensor -- and dimmer than this.
        ///
        /// The threshold is in whatever units the driver publishes intensity in,
        /// and drivers disagree: this one normalises to [0, 1], which is what
        /// Patchwork++'s default assumes, while others report 0-255. Check before
        /// changing it. Set too high it does not filter noise, it deletes every
        /// steeply-downward point there is, which is the whole near field.
        double rnr_intensity_thr = 0.2;

        // Patchwork++'s own knobs, at its defaults. They are here because ground
        // removal is the stage whose mistakes this node cannot recover from, and
        // leaving them compiled in means the only way to tune it is to rebuild.
        /// A patch with fewer points than this is not fitted at all -- Patchwork++
        /// puts the whole patch into non-ground and moves on. It is the direct
        /// cause of surviving ground patches at the edge of the field of view and
        /// in far rings, and on a sensor that does not sweep 360 degrees there are
        /// more thinly-populated patches than the default was chosen for.
        int ground_num_min_pts = 10;
        /// How thick the ground is allowed to be: a point further than this from
        /// the fitted patch plane is not ground. The one to raise on rough asphalt
        /// and to lower on a sensor precise enough to resolve the difference.
        double ground_thickness = 0.125;
        /// Seed band for the initial plane fit, and how many of the lowest points
        /// seed it.
        double ground_seed_margin = 0.125;
        int ground_num_lpr = 20;
        int ground_num_iter = 3;
        /// How far a patch normal may lean from vertical and still be ground,
        /// as the cosine of the angle. 0.707 is 45 degrees.
        double ground_uprightness_thr = 0.707;
        /// Region-wise vertical plane fitting, and temporal ground revert. Both
        /// are Patchwork++ over Patchwork and both default on.
        bool ground_vertical_plane_fit = true;
        bool ground_temporal_revert = true;
        /// Concentric Zone Model geometry: sectors and rings per zone, innermost
        /// first. Four entries each -- Patchwork++ indexes all four unconditionally
        /// whatever num_zones says.
        ///
        /// Sectors divide the full circle, so a narrow field of view does not make
        /// them coarser, it just leaves most of them empty. What justifies going
        /// finer than upstream's {16, 32, 54, 32} is point density: those were
        /// drawn for a 64-beam spinning sensor, and a patch only gets fitted at all
        /// if it holds ground_num_min_pts points. A 0.2 degree sensor puts enough
        /// in a third of the area, and smaller patches follow the surface better.
        std::vector<int> ground_sectors_each_zone = {48, 96, 162, 96};
        std::vector<int> ground_rings_each_zone = {2, 4, 4, 4};

        // -- Track surface ------------------------------------------------------
        /// Cell size of the height map built from the points Patchwork++ called
        /// ground. Every height below is measured against that map rather than
        /// against one flat plane at sensor_height, which is what lets the filter
        /// survive a mount whose real tilt does not match its extrinsic, and a
        /// track that is not level. Cells with no ground return fall back to the
        /// nearest one that has, and then to the flat plane.
        double ground_cell_size = 1.0;
        /// How far a cell of that map may sit from the flat plane at
        /// sensor_height before it is not believed.
        ///
        /// Ground removal is not immune to below-floor multipath: a return from
        /// under the track can be handed back as *ground*, and one such cell drags
        /// the surface down several metres. Everything is measured against that
        /// surface, so the rest of the noise down there then sits a plausible
        /// cone's height above it and is published as a cone. The prior is only
        /// wrong by however much the track really rises and falls, so this can be
        /// generous and still catch a five metre hole.
        double ground_max_deviation = 1.0;

        // -- Elevation band -----------------------------------------------------
        // Heights above the estimated track surface, applied per point before
        // clustering. Patchwork++ decides what is ground; this decides what is at
        // a height a cone could be at, which is a different question and the one
        // that catches what ground removal gets wrong.
        /// Floor. Below-floor multipath returns -- the asphalt reflecting a ray
        /// into something and back -- come out metres under the track and would
        /// otherwise cluster happily, and the ground patches Patchwork++ leaves
        /// behind sit right at the surface, flat and wide enough to pass the cone
        /// filter that a 2 cm min_height lets through. Both are below this.
        double min_elevation = 0.05;
        /// Ceiling. Has to stay above max_height, and that is what keeps it safe:
        /// anything reaching this is by definition taller than a cone, so a wall,
        /// a barrier or a marshal is still rejected as too tall rather than being
        /// cut down into something cone-shaped. Mostly it is what stops the whole
        /// scene above the cones from being clustered at all.
        double max_elevation = 1.00;

        // -- Ego return rejection -----------------------------------------------
        // Half-extents of a box centred on the car's origin, in the car frame.
        // Points inside it are the car looking at itself -- nosecone, bodywork,
        // driver -- and on a hoop-mounted sensor they are a large, permanent,
        // cone-height slab a few tens of centimetres away, so they both cost most
        // of the clustering time and pass the cone filter. There is nothing to be
        // lost here: this sensor cannot see the track surface closer than about
        // sensor_height / tan(half the vertical FOV) anyway.
        double ego_half_length = 1.8;
        double ego_half_width = 1.0;

        // -- Clustering (PCL Euclidean extraction) ------------------------------
        /// Voxel edge to downsample to before clustering; 0 turns it off.
        ///
        /// This is the one number that decides how long a frame takes. Clustering
        /// costs one radius search per point and the work in each is the number of
        /// neighbours it returns, so it goes with point *density*, not point count.
        /// A 0.2 degree sensor puts points about a centimetre apart at 3 m, which
        /// puts thousands inside a 30 cm radius, and the near field alone then
        /// costs more than the whole rest of the pipeline.
        ///
        /// Keep it well under a cone's width so a cone still arrives as several
        /// voxels. What it does change is point counts, so min_cluster_points and
        /// max_cluster_points are counts of voxels, not of returns.
        double voxel_leaf = 0.05;
        /// Neighbourhood radius: two points join the same cluster when they sit
        /// within this of each other. It has to exceed the vertical ring spacing on
        /// a cone at ground_max_range, or a far cone breaks into one cluster per
        /// ring.
        double cluster_tolerance = 0.30;

        // -- Cone acceptance ----------------------------------------------------
        // Bounds on a cluster's axis-aligned box, in the car frame, with heights
        // measured from the track surface. The defaults bracket both Formula
        // Student cone sizes (small 228 x 228 x 325 mm, big orange 285 x 285 x
        // 505 mm) with margin for how sparsely a LiDAR actually samples them.
        double max_width = 0.50;
        /// How far a cluster has to reach above the track to be a cone.
        ///
        /// The temptation is to set this near zero, on the reasoning that a far
        /// cone is caught by a single scan ring and so has no measurable height.
        /// That is true of a 16- or 32-beam spinning sensor and false here: 0.2
        /// degrees is 8.7 cm at 25 m, so a 325 mm cone still spans three or four
        /// rows, and measured cones on this sensor reach 0.16-0.26 m of height at
        /// every range out to 30 m. A floor near zero buys nothing and admits
        /// every flat thing on the track -- a ground patch ground removal missed,
        /// a kerb edge, a seam -- which is exactly the shape of the false
        /// positives it lets through. Re-measure it if the sensor changes.
        double min_height = 0.12;
        double max_height = 0.65;
        /// Height above the track that separates a big orange cone from a small
        /// one. The two stand 505 mm and 325 mm tall, and a 0.2 degree sensor
        /// samples 35 mm apart vertically at 10 m, so the gap is many times the
        /// measurement -- this is the one classification a colourless detector
        /// can make honestly. It matters because the four big cones on the start
        /// line are what the unknown-track planner roots the track on.
        ///
        /// Sits nearer the small cone than halfway on purpose: a cluster is the
        /// part of a cone the sensor happened to catch, so its top under-reads
        /// the real apex far more often than it over-reads it.
        double big_cone_min_height = 0.40;
        /// How far the bottom of a cluster may sit off the estimated ground plane.
        /// Signed both ways: below means leftover ground, above means the cluster
        /// is floating and is not resting on the track.
        double max_base_offset = 0.25;
        /// The point-count band a cluster has to fall in. Enforced by the
        /// extraction itself rather than by a filter afterwards, so a wall or a
        /// spectator never gets accumulated only to be thrown away.
        int min_cluster_points = 2;
        int max_cluster_points = 400;

        // -- Expected return count ----------------------------------------------
        // How many returns a cone *should* give at the range it was found at. The
        // count band above is a pair of constants and so cannot say anything: at
        // 3 m a cone gives ~57 returns and at 25 m ~3, and [2, 400] admits both
        // and everything between. Range is known per cluster, so the count can be
        // checked against what the sensor can physically put on a cone that far
        // away -- which is the strongest single discriminator in the data.
        //
        // Counted on the returns, not the voxels: downsampling flattens the curve
        // from 57:3 to 17:2 and by 15 m leaves nothing to test against.
        /// Returns expected on a cone at 10 m, and how the count falls with range
        /// as an exponent, so expected = at_10m * (10 / range) ^ exponent. Both
        /// are properties of a sensor rather than of a track: fit them once by
        /// logging cluster counts against range. The 1.65 here is measured, not
        /// the 2.0 an inverse-square argument would predict, because the returns
        /// on a cone are not an area sample of it.
        double expected_points_at_10m = 10.0;
        double expected_points_exponent = 1.65;
        /// Multiplicative slack either side. Wide on purpose: this is here to
        /// reject a cluster that cannot be a cone at that range, not to fit one.
        /// The upper bound has to clear a big orange cone, which has about 2.4
        /// times the frontal area of a small one. Zero disables the test.
        double expected_points_min_ratio = 0.33;
        double expected_points_max_ratio = 4.0;

        // -- How well a detection is located ------------------------------------
        //
        // A LiDAR does not know a cone equally well in every direction. Along the
        // beam it measures a range directly and gets it right to a few
        // centimetres whatever the distance. Across the beam it has only the
        // angle the beam was pointing at, so the error there is that angle times
        // the range -- small nearby, and the dominant term by far at the end of
        // the field of view.
        //
        // Reporting that honestly is what lets SLAM stop treating a cone at 25 m
        // like one at 3 m. It matters more than it sounds: a distant cone has the
        // longest lever arm on heading, so over-trusting one turns its own
        // measurement error into a pose correction.

        /// Radial (along-beam) standard deviation, metres. Roughly the sensor's
        /// own range accuracy, plus the fact that only the near face of the cone
        /// is ever seen, so the centroid sits a little short of the true centre.
        double range_sigma = 0.05;
        /// Angular standard deviation, radians, turned into a cross-beam
        /// distance by multiplying by range. Rather more than one beam width: it
        /// stands for where on the cone the returns happened to land, not for the
        /// sensor's pointing precision.
        double bearing_sigma = 0.006;
        /// How far apart the points of one cluster are spread, metres. Averaging
        /// n of them locates the centroid to about this over sqrt(n), which is
        /// what makes a three-point cluster at range properly uncertain and a
        /// fifty-point cluster up close properly trusted.
        double cluster_spread = 0.15;
        /// Rejects the barriers, marshals and spectators that flank the track.
        double max_lateral_offset = 10.0;
    };

    /// One accepted cone, positioned at its cluster centroid in the car frame.
    struct Detection
    {
        Eigen::Vector3f position;
        int num_points;
        /// Tall enough to be one of the big orange cones on the start line.
        /// Colour is not measured here; height is.
        bool big;
        /// How well this position is known, as a 2x2 row-major covariance in the
        /// sensor frame: [xx, xy, yx, yy]. See ConeDetectorParams for the model.
        std::array<double, 4> covariance;
    };

    /// Colourless, geometry-only cone detector: Patchwork++ takes the ground out,
    /// PCL's Euclidean cluster extraction cuts what is left into clusters, and a
    /// size-and-placement filter keeps the cone-shaped ones.
    class ConeDetector
    {
    public:
        using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

        explicit ConeDetector(const ConeDetectorParams &params);
        ~ConeDetector();

        /// `cloud_lidar` holds one point per row as x, y, z and optionally
        /// intensity, in the LiDAR's own frame. `car_from_lidar` places those
        /// points in the car frame, which is where every cone heuristic lives,
        /// since each one is a statement about the track rather than the sensor.
        /// It is applied in two halves -- rotation before ground removal, offset
        /// after -- so that Patchwork++ sees the levelled cloud its ground model
        /// assumes even when the sensor is mounted tilted.
        ///
        /// The result and the debug views below are owned by the detector and stay
        /// valid until the next call.
        const std::vector<Detection> &detect(const Eigen::MatrixXf &cloud_lidar,
                                             const Eigen::Isometry3f &car_from_lidar);

        /// Non-ground points in the car frame, and the cluster each one fell into:
        /// the index of the accepted cone, or -1 for points the extraction left
        /// unclustered and for clusters the filter threw away.
        const PointCloud &nonground() const { return *nonground_; }
        const std::vector<int> &labels() const { return labels_; }

        /// Wall time of each stage of the last detect(), in milliseconds. What
        /// dominates moves with the scene, so this is worth having on the wire.
        struct StageTimes
        {
            double level_ms = 0.0;
            double patchwork_ms = 0.0;
            double ground_map_ms = 0.0;
            double band_ms = 0.0;
            double kdtree_ms = 0.0;
            double cluster_ms = 0.0;
            double accept_ms = 0.0;
        };
        const StageTimes &stage_times() const { return stage_; }

        /// Wall time of the last detect(), split at the stage that dominates it.
        double ground_removal_ms() const { return ground_removal_ms_; }
        double total_ms() const { return total_ms_; }

    private:
        struct ClusterBounds;
        struct GroundMap;

        bool is_cone(const ClusterBounds &cluster) const;
        /// Height of a cluster above the track under it.
        float height_above_ground(const ClusterBounds &cluster) const;
        std::array<double, 4> position_covariance(const Eigen::Vector3f &centroid, int num_points) const;

        /// Where the LiDAR sits in the car frame, so a detection's bearing can be
        /// measured from the sensor that saw it.
        Eigen::Vector2f sensor_xy_ = Eigen::Vector2f::Zero();
        /// Returns inside a cluster's box, counted on the cloud as it was before
        /// downsampling, against what the sensor could put on a cone that far off.
        bool has_plausible_return_count(const ClusterBounds &cluster) const;

        ConeDetectorParams params_;
        // Built on the first cloud rather than here, because whether
        // reflected-noise removal can run is a property of the cloud.
        std::unique_ptr<patchwork::PatchWorkpp> patchwork_;

        std::vector<Detection> detections_;
        // The input cloud rotated level, held across frames so a sweep-sized
        // buffer is allocated once rather than every 100 ms.
        Eigen::MatrixXf level_;
        // The track surface under the current cloud. Same reason for living here.
        std::unique_ptr<GroundMap> ground_;
        PointCloud::Ptr nonground_;
        // Downsampling target and the filter itself, both held across frames so
        // the per-cloud cost is the filtering rather than the allocation.
        PointCloud::Ptr downsampled_;
        // The banded cloud before downsampling. Points into whichever of the two
        // above holds it, so nothing is copied to keep it.
        const PointCloud *returns_ = nullptr;
        pcl::VoxelGrid<pcl::PointXYZ> voxel_;
        // Held across frames so the per-cloud cost is rebuilding the index, not
        // also allocating the object that owns it.
        pcl::search::KdTree<pcl::PointXYZ>::Ptr search_;
        std::vector<int> labels_;

        StageTimes stage_;
        double ground_removal_ms_ = 0.0;
        double total_ms_ = 0.0;
    };

} // namespace mirena

#endif // MIRENA_PERCEPTION_CONE_DETECTOR_HPP
