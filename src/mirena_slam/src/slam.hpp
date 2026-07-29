#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/linear/NoiseModel.h> // Ensure NoiseModel is included for robust_meas_noise_
#include <map>
#include <string>
#include <vector>

#define ODOM_NOISE_DEFAULT gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.02))
// SOTA Robust Kernel: Dynamic Covariance Scaling (DCS)
// A phi threshold of 1.0 sharply cuts off outliers. If a hypothesis is wrong,
// DCS will weight its factor to ~0.0 during optimization.
#define MEAS_NOISE_DEFAULT  gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.1, 0.1))

namespace mirena
{

    struct Landmark
    {
        double x, y;
        std::string color;
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

    private:
        void update_isam();

        /**
         * @brief Performs Multi-Hypothesis Data Association
         * Returns a list of all mathematically viable matches for each observation
         * using Mahalanobis Distance and iSAM2 marginals.
         */
        std::vector<std::vector<int>> find_candidate_associations(const std::vector<Landmark> &observations, gtsam::SharedNoiseModel meas_noise);

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
        std::map<int, std::string> landmark_colors_;
        std::map<int, double> landmark_confidence_;

        double last_update_elapsed_ms_ = 0;

        // Increased threshold for Multi-Hypothesis gating.
        // 9.0 is roughly a 99% confidence interval for 2 Degrees of Freedom (x,y).
        const double mahalanobis_threshold_ = 5.98;
        double max_confidence_yet = 0;
    };

} // namespace mirena