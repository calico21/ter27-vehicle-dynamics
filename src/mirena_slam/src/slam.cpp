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
std::vector<std::vector<int>> Slam::find_candidate_associations(
    const std::vector<Landmark>& observations, 
    gtsam::SharedNoiseModel meas_noise) 
{
    int num_obs = observations.size();
    std::vector<std::vector<int>> all_candidates(num_obs);
    if (num_obs == 0) return all_candidates;

    // -------------------------------------------------------------
    // 1. Safe Vehicle Pose Retrieval (Checks new_values_ AND current_estimate_)
    // -------------------------------------------------------------
    gtsam::Symbol curr_x_key('x', step_count_);
    gtsam::Symbol prev_x_key('x', step_count_ > 0 ? step_count_ - 1 : 0);
    gtsam::Pose2 vehicle_pose(0.0, 0.0, 0.0);

    try {
        if (new_values_.exists(curr_x_key)) {
            vehicle_pose = new_values_.at<gtsam::Pose2>(curr_x_key);
        } else if (current_estimate_.exists(curr_x_key)) {
            vehicle_pose = current_estimate_.at<gtsam::Pose2>(curr_x_key);
        } else if (new_values_.exists(prev_x_key)) {
            vehicle_pose = new_values_.at<gtsam::Pose2>(prev_x_key);
        } else if (current_estimate_.exists(prev_x_key)) {
            vehicle_pose = current_estimate_.at<gtsam::Pose2>(prev_x_key);
        }
    } catch (const std::exception& e) {
        // Fallback to origin if state lookup fails on step 0
        vehicle_pose = gtsam::Pose2(0.0, 0.0, 0.0);
    }

    // Safe Pose Covariance Retrieval
    gtsam::Matrix33 P_pose = gtsam::Matrix3::Identity() * 0.05;
    try {
        if (step_count_ > 0) {
            P_pose = isam_.marginalCovariance(prev_x_key);
        }
    } catch (...) {
        P_pose = gtsam::Matrix3::Identity() * 0.05;
    }

    // -------------------------------------------------------------
    // 2. Fetch Local Existing Landmarks (From both estimates)
    // -------------------------------------------------------------
    std::vector<int> existing_ids;
    gtsam::Point2 vehicle_pos = vehicle_pose.translation();
    
    for (auto const& [id, color] : landmark_colors_) {
        gtsam::Symbol l_key('l', id);
        gtsam::Point2 lm_pos;
        
        if (new_values_.exists(l_key)) {
            lm_pos = new_values_.at<gtsam::Point2>(l_key);
        } else if (current_estimate_.exists(l_key)) {
            lm_pos = current_estimate_.at<gtsam::Point2>(l_key);
        } else {
            continue; // Landmark key not initialized yet
        }

        // 20-meter local radius filter
        if ((lm_pos - vehicle_pos).norm() < 20.0) {
            existing_ids.push_back(id);
        }
    }

    // If no local landmarks exist, return empty vectors to trigger creation of new landmarks
    if (existing_ids.empty()) return all_candidates;

    // -------------------------------------------------------------
    // 3. Precompute Landmark Covariances
    // -------------------------------------------------------------
    std::map<int, gtsam::Matrix22> precomputed_covs;
    for (int id : existing_ids) {
        try { 
            precomputed_covs[id] = isam_.marginalCovariance(gtsam::Symbol('l', id)); 
        } catch (...) { 
            precomputed_covs[id] = gtsam::Matrix2::Identity() * 0.1; 
        }
    }

    gtsam::Matrix22 R = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(meas_noise)->covariance();

    // -------------------------------------------------------------
    // 4. Data Association Loop
    // -------------------------------------------------------------
    for (int i = 0; i < num_obs; ++i) {
        gtsam::Point2 measured_pt(observations[i].x, observations[i].y);
        
        double best_d2 = std::numeric_limits<double>::infinity();
        int best_lm_id = -1;

        for (int lm_id : existing_ids) {
            gtsam::Symbol l_key('l', lm_id);
            gtsam::Point2 lm_pos = new_values_.exists(l_key) ? 
                                    new_values_.at<gtsam::Point2>(l_key) : 
                                    current_estimate_.at<gtsam::Point2>(l_key);

            // COLOR GATING: Soft color matching
            bool obs_is_orange = (observations[i].color.find("orange") != std::string::npos);
            bool lm_is_orange  = (landmark_colors_[lm_id].find("orange") != std::string::npos);

            if (landmark_confidence_[lm_id] > 0.8 && observations[i].confidence > 0.8) {
                if ((observations[i].color != landmark_colors_[lm_id]) && !obs_is_orange && !lm_is_orange) {
                    continue; 
                }
            }

            // Jacobians for transformTo (World -> Vehicle frame)
            gtsam::Matrix23 H_pose;
            gtsam::Matrix22 H_lm;
            gtsam::Point2 predicted_pt = vehicle_pose.transformTo(lm_pos, H_pose, H_lm);
            
            gtsam::Vector2 innovation = measured_pt - predicted_pt;

            // Spatial Euclidean pre-filter
            if (innovation.norm() > 2.0) continue;

            // Full Innovation Covariance: S = H_pose * P_pose * H_pose^T + H_lm * P_lm * H_lm^T + R
            gtsam::Matrix22 P_lm = precomputed_covs[lm_id];
            gtsam::Matrix22 S = H_pose * P_pose * H_pose.transpose() + 
                                H_lm * P_lm * H_lm.transpose() + R;

            double d2 = innovation.transpose() * S.inverse() * innovation;

            if (d2 < mahalanobis_threshold_ && d2 < best_d2) {
                best_d2 = d2;
                best_lm_id = lm_id;
            }
        }

        // Push ONLY the best match for this observation
        if (best_lm_id != -1) {
            all_candidates[i].push_back(best_lm_id);
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