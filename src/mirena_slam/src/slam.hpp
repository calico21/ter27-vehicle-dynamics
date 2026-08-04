#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/linear/NoiseModel.h>
#include <array>
#include <map>
#include <string>
#include <vector>

#define ODOM_NOISE_DEFAULT gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.02))
#define MEAS_NOISE_DEFAULT gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.1, 0.1))

namespace mirena
{

    struct Landmark
    {
        double x, y;
        /// What perception could tell about the cone without reading its colour,
        /// which in a colourless pipeline is its size: `cone_big_orange` for the
        /// cones marking the start line, `unknown_cone` for the rest.
        std::string type;
        double confidence;
        std::array<double, 4> covariance; // [00, 01, 10, 11] -> [xx, xy, yx, yy]
    };

    class Slam
    {
    public:
        Slam(const gtsam::Pose2 &initial_raw_pose);
        virtual ~Slam() = default;

        /**
         * @brief Adds a new pose and associated landmark observations
         * @param raw_odom The absolute raw odometry pose (delta is computed internally)
         * @param obs Local coordinates of perceived landmarks
         */
        void add_keyframe(const gtsam::Pose2 &raw_odom, const std::vector<Landmark> &obs, gtsam::SharedNoiseModel odom_noise = ODOM_NOISE_DEFAULT, gtsam::SharedNoiseModel meas_noise = MEAS_NOISE_DEFAULT);

        const std::vector<Landmark>& get_map() const;
        gtsam::Pose2 get_latest_pose() const;
        gtsam::Pose2 get_correction() const;
        double get_last_update_elapsed_ms() const;

        /// Observations paired with an existing landmark on the last keyframe,
        /// and detections that became new ones. Once the track has been driven
        /// the new count should sit at zero; if it does not, association is
        /// failing to recognise cones already mapped.
        int get_last_matched_count() const;
        int get_last_new_count() const;

    private:
        void update_isam();

        /**
         * @brief Nearest-neighbour data association, scored by Mahalanobis
         *        distance and solved as an optimal one-to-one assignment.
         *
         * Cost is the squared Mahalanobis distance of the innovation, gated at
         * `mahalanobis_threshold_`, with `max_association_distance_` as a hard
         * Euclidean backstop. The innovation covariance is
         *
         *   S = H_x P_pose H_x^T + R^T P_lm R + R_meas,
         *   H_x = [ -R^T | R^T J (l - p) ],  J = [[0, 1], [-1, 0]]
         *
         * so the gate widens exactly where the estimate is genuinely uncertain.
         * The heading column is not a refinement: at 10 m a yaw sigma of
         * 0.05 rad is half a metre of lateral error, which dominates everything
         * else in S, and dropping it leaves the gate far too tight at range.
         *
         * The assignment is the other half. Scoring each observation
         * independently lets two cones detected in one frame both claim a single
         * landmark, which puts two contradictory measurement factors on one point
         * and drags it to the midpoint of two real cones.
         *
         * @return association[i] is the landmark id matched to observations[i],
         *         or -1 for none.
         */
        std::vector<int> associate(const gtsam::Pose2 &pred_pose,
                                   const gtsam::Matrix3 &pose_cov_map,
                                   const std::vector<Landmark> &observations,
                                   const gtsam::SharedNoiseModel &meas_noise);

        /// Accumulated type evidence for one landmark: how often each type was
        /// reported, and which won.
        ///
        /// Unweighted, because a colourless detector has nothing graded to say --
        /// a cluster is tall enough to be a big cone or it is not, and it reports
        /// the same confidence either way. Counting is also what makes a single
        /// misread cost one vote rather than latching the landmark, which matters
        /// because the big cones are what the planner roots the track on.
        struct TypeVote
        {
            std::map<std::string, int> counts;
            std::string best;
            int best_count = 0;
            int total = 0;

            void add(const std::string &type);
            double confidence() const
            {
                return total > 0 ? static_cast<double>(best_count) / total : 0.0;
            }
        };

        struct LandmarkRecord
        {
            gtsam::Point2 position;
            /// Marginal covariance, refreshed on a period rather than on every
            /// keyframe: each one costs a back-substitution over the Bayes tree,
            /// and a converged cone's covariance barely moves. Doing every
            /// landmark every keyframe is what made the original dominate the
            /// update budget.
            gtsam::Matrix2 covariance = gtsam::Matrix2::Identity() * 0.1;
            long cov_step = -1;
            TypeVote types;
        };

        gtsam::ISAM2 isam_;
        gtsam::Values current_estimate_;
        gtsam::NonlinearFactorGraph new_factors_;
        gtsam::Values new_values_;

        // Odom correction
        std::vector<Landmark> last_map_;
        gtsam::Pose2 last_raw_odom_;
        gtsam::Pose2 map_to_odom_correction_;

        int step_count_ = 0;
        int landmark_count_ = 0;
        std::map<int, LandmarkRecord> landmarks_;

        /// Covariance of the latest pose in its own tangent (body) frame, which is
        /// how GTSAM reports Pose2 uncertainty. One marginal per keyframe.
        gtsam::Matrix3 pose_cov_ = gtsam::Matrix3::Identity() * 1e-12;

        double last_update_elapsed_ms_ = 0;
        int last_matched_count_ = 0;
        int last_new_count_ = 0;

        /// Chi-square gate on the squared Mahalanobis distance, and the knob that
        /// actually decides associations. 5.99 is the 95th percentile for 2
        /// degrees of freedom (x, y); 9.21 is the 99th.
        ///
        /// What it permits is not a fixed radius, which is the point: with the
        /// floor below, S is about 0.11 m^2 for a well constrained cone close by,
        /// so the gate reaches roughly 0.8 m; at 10 m a yaw sigma of 0.05 rad adds
        /// 0.25 m^2 through the heading term and it opens to about 1.5 m. Tight
        /// where the estimate is good, generous where it genuinely is not.
        const double mahalanobis_threshold_ = 5.99;

        /// Hard Euclidean backstop, metres, applied before the chi-square test.
        ///
        /// Sized so it does *not* bind in normal operation -- it exists only to
        /// stop a pose that has gone badly wrong from reaching clean across the
        /// track, whatever its covariance claims. Set below the gate radii above
        /// it becomes the real gate and the Mahalanobis test stops mattering.
        const double max_association_distance_ = 2.5;

        /// Nothing new is created within this of a cone already mapped. Formula
        /// Student cones stand metres apart, so anything closer is a second
        /// report of a cone the frame already has, or a re-detection the
        /// assignment could not claim -- either way a cone there would stack on
        /// top of an existing one. Kept independent of the gate so the guarantee
        /// holds however the gate is tuned.
        const double min_cone_separation_ = 1.0;

        /// Added to a landmark's covariance before gating. A converged cone
        /// otherwise carries a gate a couple of centimetres wide, and every
        /// re-detection that misses it becomes a duplicate sitting on top of it.
        const double covariance_floor_ = 0.09;

        const int marginal_refresh_period_ = 10;
        const double local_map_radius_ = 20.0;
    };

} // namespace mirena
