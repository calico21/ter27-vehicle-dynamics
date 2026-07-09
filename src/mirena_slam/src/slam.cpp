#include "slam.hpp"
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/BetweenFactor.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <limits>
#include <chrono>

using namespace std;

namespace mirena
{

    Slam::Slam(const gtsam::Pose2 &initial_raw_pose)
    {
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip = 1;
        
        // CRITICAL FOR SOTA: Enables ISAM2 to dynamically evaluate the error for 
        // the DCS robust kernel, allowing it to "shut off" bad multi-hypothesis factors.
        params.evaluateNonlinearError = true; 
        isam_ = gtsam::ISAM2(params);

        // Initialize the Odometry
        last_raw_odom_ = initial_raw_pose;
        map_to_odom_correction_ = gtsam::Pose2(); // Identity (0,0,0)


        // Initial Prior
        gtsam::Symbol x0('x', 0);
        new_factors_.addPrior(x0, gtsam::Pose2(0, 0, 0), gtsam::noiseModel::Isotropic::Sigma(3, 1e-6));
        new_values_.insert(x0, gtsam::Pose2(0, 0, 0));
        update_isam();
    }

void Slam::add_keyframe(const gtsam::Pose2 &raw_odom, 
                        const std::vector<Landmark> &obs, 
                        gtsam::SharedNoiseModel odom_noise, 
                        gtsam::SharedNoiseModel meas_noise)
{
    step_count_++;
    gtsam::Symbol curr_x('x', step_count_);
    gtsam::Symbol prev_x('x', step_count_ - 1);

    // 1. Odometry Delta
    gtsam::Pose2 delta = last_raw_odom_.between(raw_odom);
    last_raw_odom_ = raw_odom;

    // 2. Add Odometry Factor
    new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(prev_x, curr_x, delta, odom_noise);

    // 3. Initial Prediction for current pose
    gtsam::Pose2 pred_pose = current_estimate_.at<gtsam::Pose2>(prev_x).compose(delta);
    new_values_.insert(curr_x, pred_pose);

    // 4. Noise Models: Make Big Orange "Stiff"
    // Standard noise (meas_noise) is usually ~0.1m. 
    // We set Orange to 0.02m to prevent landmarks from "bleeding" into each other.
    auto orange_noise = gtsam::noiseModel::Isotropic::Sigma(2, 0.02);

    // 5. Data Association
    std::vector<std::vector<int>> hypotheses_list = find_candidate_associations(obs, meas_noise);

    for (size_t i = 0; i < obs.size(); ++i)
    {
        const auto &o = obs[i];
        const auto &candidates = hypotheses_list[i];
        assert(candidates.size() <= 1);
        gtsam::Expression<gtsam::Pose2> x_(curr_x);
        
        bool is_orange = (o.color == "big_orange");
        auto active_noise = is_orange ? orange_noise : meas_noise;

        if (candidates.empty())
        {
            // BRAND NEW LANDMARK
            int new_id = landmark_count_++;
            gtsam::Symbol l_key('l', new_id);
            
            gtsam::Point2 global_pt = pred_pose.transformFrom(gtsam::Point2(o.x, o.y));
            new_values_.insert(l_key, global_pt);
            landmark_colors_[new_id] = o.color;
            landmark_confidence_[new_id] = o.confidence;

            gtsam::Expression<gtsam::Point2> l_(l_key);
            
            // For a brand new landmark, we use DCS with the stiff noise.
            new_factors_.add(gtsam::ExpressionFactor<gtsam::Point2>(
                gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::DCS::Create(1.0), active_noise),
                gtsam::Point2(o.x, o.y), gtsam::transformTo(x_, l_)));
        }
        else
        {
            // MULTI-HYPOTHESIS ASSOCIATION
            for (int id : candidates)
            {
                gtsam::Symbol l_key('l', id);
                
                // Update Metadata if existing
                if (landmark_colors_.count(id) && landmark_colors_[id] != o.color)
                {
                    if (landmark_confidence_[id] < o.confidence)
                    {
                        landmark_colors_[id] = o.color;
                        landmark_confidence_[id] = o.confidence; 
                    }
                }

                gtsam::Expression<gtsam::Point2> l_(l_key);
                
                // DCS evaluates every hypothesis. Because 'orange_noise' is so stiff,
                // ISAM2 will quickly realize which L_key is the correct match and
                // the DCS kernel will down-weight the other 3 "ghost" associations.
                new_factors_.add(gtsam::ExpressionFactor<gtsam::Point2>(
                    gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::DCS::Create(1.0), active_noise),
                    gtsam::Point2(o.x, o.y), gtsam::transformTo(x_, l_)));
            }
        }
    }

    update_isam();
}
 std::vector<std::vector<int>> Slam::find_candidate_associations(const std::vector<Landmark>& observations, gtsam::SharedNoiseModel meas_noise) 
    {

        int num_obs = observations.size();
        std::vector<std::vector<int>> all_candidates(num_obs);
        if (num_obs == 0) return all_candidates;

        // 1. Get current pose
        gtsam::Symbol curr_x_key('x', step_count_);
        gtsam::Symbol prev_x_key('x', step_count_ - 1);
        
        gtsam::Pose2 vehicle_pose = new_values_.exists(curr_x_key) ? 
                                    new_values_.at<gtsam::Pose2>(curr_x_key) : 
                                    current_estimate_.at<gtsam::Pose2>(prev_x_key);

        // 2. Local Map Fetch (Radius 20m)
        std::vector<int> existing_ids;
        gtsam::Point2 vehicle_pos = vehicle_pose.translation();
        for (auto const& [id, color] : landmark_colors_) {
            gtsam::Symbol l_key('l', id);
            if (!current_estimate_.exists(l_key)) continue;
            if ((current_estimate_.at<gtsam::Point2>(l_key) - vehicle_pos).norm() < 20.0) {
                existing_ids.push_back(id);
            }
        }

        if (existing_ids.empty()) return all_candidates; // Return empty vectors -> trigger new landmarks

        // 3. Precompute Covariances for speed
        std::map<int, gtsam::Matrix22> precomputed_covs;
        for (int id : existing_ids) {
            try { precomputed_covs[id] = isam_.marginalCovariance(gtsam::Symbol('l', id)); }
            catch (...) { precomputed_covs[id] = gtsam::Matrix2::Identity() * 0.1; }
        }

        gtsam::Matrix22 P_pos = gtsam::Matrix2::Identity() * 0.1;
        try { P_pos = isam_.marginalCovariance(prev_x_key).block<2, 2>(0, 0); } catch (...) {}
        gtsam::Matrix22 R = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(meas_noise)->covariance();
        gtsam::Matrix22 RiT = vehicle_pose.rotation().matrix().transpose();

        // 4. Find all matching hypotheses per observation

        double min_d2_yet;
        for (int i = 0; i < num_obs; ++i) {
            min_d2_yet = std::numeric_limits<double>::infinity();
            gtsam::Point2 measured_pt(observations[i].x, observations[i].y);
            for (int lm_id : existing_ids) {
                // HARD COLOR GATE: Prune impossible matches immediately
                if (observations[i].color != landmark_colors_[lm_id] && 
                   (observations[i].color.find("orange") == std::string::npos && 
                    landmark_colors_[lm_id].find("orange") == std::string::npos)) continue;
                
                gtsam::Point2 lm_pos = current_estimate_.at<gtsam::Point2>(gtsam::Symbol('l', lm_id));
                gtsam::Vector2 innovation = measured_pt - vehicle_pose.transformTo(lm_pos);
                //Cmputamos covarianza
                gtsam::Matrix22 P_lm = precomputed_covs[lm_id];
                const double COV_FLOOR = 0.01;
                P_lm(0,0) += COV_FLOOR;
                P_lm(1,1) += COV_FLOOR;
                // EUCLIDEAN PRE-GATE: Skip heavy math if it's obviously wrong (>2.5m away)
                if (innovation.norm() > 2.5) continue;

                // MAHALANOBIS DISTANCE
                gtsam::Matrix22 S = RiT * (P_pos + P_lm) * vehicle_pose.rotation().matrix() + R;
                double d2 = innovation.transpose() * S.inverse() * innovation;
                
                // MULTI-HYPOTHESIS TRIGGER: If it passes the threshold, keep it!
                // Note: You can slightly increase mahalanobis_threshold_ in your config
                // to allow the graph to resolve more ambiguity safely.
                if (d2 < mahalanobis_threshold_ && d2 < min_d2_yet) {
                    min_d2_yet = d2;
                    all_candidates.at(i).clear();
                    all_candidates.at(i).push_back(lm_id);
                }
            }
        }

        return all_candidates;
    }
    void Slam::update_isam()
    {
        using namespace std::chrono;
        auto start_time = high_resolution_clock::now();

        // 1. Record graph state before update
        size_t num_factors_new = new_factors_.size();
        size_t num_values_new = new_values_.size();

        // 2. Perform the ISAM2 update
        // Note: isam_.update() returns an ISAM2Result object containing 
        // valuable info about linear variables, error, etc.
        gtsam::ISAM2Result result = isam_.update(new_factors_, new_values_);
        
        // 3. Calculate Estimate (This gets slower as the graph grows)
        current_estimate_ = isam_.calculateEstimate();

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time).count();

        // 4. Extract optimized pose
        gtsam::Symbol curr_x('x', step_count_);
        gtsam::Pose2 optimized_pose = current_estimate_.at<gtsam::Pose2>(curr_x);
        map_to_odom_correction_ = optimized_pose.compose(last_raw_odom_.inverse());

        // 5. Performance Printout
        this->last_update_elapsed_ms_ = duration / 1.0e3;
        //std::cout << "--- SLAM Step: " << step_count_ << " ---" << std::endl;
        //std::cout << "  Execution Time:  " << duration / 1000.0 << " ms" << std::endl;
        //std::cout << "  New Factors:     " << num_factors_new << std::endl;
        //std::cout << "  Total Variables: " << current_estimate_.size() << std::endl;
        //std::cout << "  Relinearized:    " << result.variablesRelinearized << " vars" << std::endl;
        
        //if (duration > 50000) { // Warning if update takes > 50ms
        //    std::cerr << "  [WARNING] SLAM Update exceeding real-time budget!" << std::endl;
        //}

        // Cleanup
        new_factors_.resize(0);
        new_values_.clear();
        
        // Generate map
        this->last_map_.clear();
        for (auto const &[id, color] : landmark_colors_)
        {
            if (current_estimate_.exists(gtsam::Symbol('l', id)))
            {
                gtsam::Point2 p = current_estimate_.at<gtsam::Point2>(gtsam::Symbol('l', id));
                this->last_map_.push_back({p.x(), p.y(), color, 1.0}); // Ensure constructor matches your Landmark struct
            }
        }
    }

    const std::vector<Landmark>& Slam::get_map() const
    {
        return this->last_map_;
    }

    gtsam::Pose2 Slam::get_latest_pose() const
    {
        return current_estimate_.at<gtsam::Pose2>(gtsam::Symbol('x', step_count_));
    }

    gtsam::Pose2 Slam::get_correction() const
    {
        return map_to_odom_correction_;
    }

    double Slam::get_last_update_elapsed_ms() const
    {
        return last_update_elapsed_ms_;
    }

} // namespace mirena