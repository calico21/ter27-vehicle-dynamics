#include "slam.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace mirena
{

    namespace
    {
        /// Cost standing in for "impossible", finite so the potentials stay
        /// well-behaved but far above anything the gate would accept.
        constexpr double kBlocked = 1e9;

        /// Maps a body-frame tangent perturbation to a map-frame one, so the pose
        /// covariance GTSAM reports can be used with map-frame Jacobians.
        gtsam::Matrix3 body_to_map(const gtsam::Pose2 &pose)
        {
            gtsam::Matrix3 A = gtsam::Matrix3::Identity();
            A.topLeftCorner<2, 2>() = pose.rotation().matrix();
            return A;
        }

        /**
         * @brief Minimum-cost assignment of rows to columns (Kuhn-Munkres /
         *        Jonker-Volgenant): shortest augmenting path with potentials,
         *        O(n^2 m). Requires at least as many columns as rows.
         * @return the column chosen for each row.
         */
        std::vector<int> hungarian(const std::vector<std::vector<double>> &cost)
        {
            const int n = static_cast<int>(cost.size());
            if (n == 0)
                return {};
            const int m = static_cast<int>(cost[0].size());

            constexpr double kInf = std::numeric_limits<double>::infinity();

            // One-based potentials and column assignment; column 0 is the
            // sentinel each augmenting path starts from.
            std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
            std::vector<int> match(m + 1, 0), path(m + 1, 0);

            for (int row = 1; row <= n; ++row)
            {
                match[0] = row;
                int col = 0;
                std::vector<double> slack(m + 1, kInf);
                std::vector<bool> done(m + 1, false);

                do
                {
                    done[col] = true;
                    const int cur = match[col];
                    double delta = kInf;
                    int next = 0;

                    for (int j = 1; j <= m; ++j)
                    {
                        if (done[j])
                            continue;
                        const double reduced = cost[cur - 1][j - 1] - u[cur] - v[j];
                        if (reduced < slack[j])
                        {
                            slack[j] = reduced;
                            path[j] = col;
                        }
                        if (slack[j] < delta)
                        {
                            delta = slack[j];
                            next = j;
                        }
                    }

                    // Shift the potentials so the chosen edge becomes tight.
                    for (int j = 0; j <= m; ++j)
                    {
                        if (done[j])
                        {
                            u[match[j]] += delta;
                            v[j] -= delta;
                        }
                        else
                        {
                            slack[j] -= delta;
                        }
                    }
                    col = next;
                } while (match[col] != 0);

                // Walk the path back, flipping the matching along it.
                do
                {
                    const int prev = path[col];
                    match[col] = match[prev];
                    col = prev;
                } while (col != 0);
            }

            std::vector<int> assignment(n, -1);
            for (int j = 1; j <= m; ++j)
                if (match[j] != 0)
                    assignment[match[j] - 1] = j - 1;
            return assignment;
        }
    } // namespace

    void Slam::TypeVote::add(const std::string &type)
    {
        if (type.empty())
            return;

        const int count = ++counts[type];
        ++total;

        if (count > best_count)
        {
            best_count = count;
            best = type;
        }
    }

    Slam::Slam(const gtsam::Pose2 &initial_raw_pose)
    {
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip = 1;
        // Deliberately off: it forces a full nonlinear error evaluation on every
        // update, and the DCS weights are recomputed at linearization anyway. It
        // buys a number for the result struct and nothing else.
        params.evaluateNonlinearError = false;
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

        // 3b. Propagate the pose covariance through the delta. Both terms live in
        //     the body tangent frame, which is where GTSAM keeps them; the
        //     associator wants it in the map frame.
        gtsam::Matrix3 odom_cov = gtsam::Matrix3::Identity() * 1e-6;
        if (auto g = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(odom_noise))
            odom_cov = g->covariance();

        const gtsam::Matrix3 adjoint = delta.inverse().AdjointMap();
        const gtsam::Matrix3 pred_cov_body = adjoint * pose_cov_ * adjoint.transpose() + odom_cov;
        const gtsam::Matrix3 A = body_to_map(pred_pose);

        // 4. Data association: one landmark per observation, one observation per
        //    landmark.
        const std::vector<int> matched =
            associate(pred_pose, A * pred_cov_body * A.transpose(), obs, meas_noise);

        last_matched_count_ = 0;
        last_new_count_ = 0;

        const gtsam::Expression<gtsam::Pose2> x_(curr_x);
        // A robust kernel on every measurement factor, so an association that
        // slipped through the gate is down-weighted rather than believed.
        const auto robust = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::DCS::Create(1.0), meas_noise);

        for (size_t i = 0; i < obs.size(); ++i)
        {
            const Landmark &o = obs[i];
            const gtsam::Point2 local(o.x, o.y);

            int id = matched[i];
            if (id < 0)
            {
                // BRAND NEW LANDMARK -- unless it lands within association reach
                // of a cone already mapped. The assignment is one-to-one, so a
                // second detection of a cone another observation just claimed
                // comes back unmatched; adding it would stack two landmarks on
                // one physical cone. Formula Student cones are metres apart, so
                // nothing this close can be a new one, and refusing costs a
                // single sighting.
                const gtsam::Point2 world = pred_pose.transformFrom(local);
                const bool crowded = std::any_of(
                    landmarks_.begin(), landmarks_.end(), [&](const auto &entry) {
                        return (entry.second.position - world).norm() < min_cone_separation_;
                    });
                if (crowded)
                    continue;

                id = landmark_count_++;
                LandmarkRecord record;
                record.position = world;
                landmarks_.emplace(id, std::move(record));
                new_values_.insert(gtsam::Symbol('l', id), world);
                ++last_new_count_;
            }
            else
            {
                ++last_matched_count_;
            }

            landmarks_.at(id).types.add(o.type);

            const gtsam::Expression<gtsam::Point2> l_(gtsam::Symbol('l', id));
            new_factors_.add(gtsam::ExpressionFactor<gtsam::Point2>(
                robust, local, gtsam::transformTo(x_, l_)));
        }

        update_isam();
    }

    std::vector<int> Slam::associate(const gtsam::Pose2 &pred_pose,
                                     const gtsam::Matrix3 &pose_cov_map,
                                     const std::vector<Landmark> &observations,
                                     const gtsam::SharedNoiseModel &meas_noise)
    {
        const size_t n = observations.size();
        std::vector<int> association(n, -1);
        if (n == 0)
            return association;

        // Local map: only what is plausibly in view. Marginals are amortised --
        // computed when a landmark first appears, then refreshed on a period.
        std::vector<int> ids;
        const gtsam::Point2 vehicle = pred_pose.translation();
        const long step = static_cast<long>(step_count_);

        for (auto &[id, record] : landmarks_)
        {
            if ((record.position - vehicle).norm() >= local_map_radius_)
                continue;

            if (record.cov_step < 0 || step - record.cov_step >= marginal_refresh_period_)
            {
                try
                {
                    record.covariance = isam_.marginalCovariance(gtsam::Symbol('l', id));
                }
                catch (const std::exception &)
                {
                    record.covariance = gtsam::Matrix2::Identity() * 0.1;
                }
                record.cov_step = step;
            }
            ids.push_back(id);
        }

        if (ids.empty())
            return association;

        const size_t m = ids.size();

        // Measurement covariance. Checked, unlike the raw dynamic_pointer_cast
        // this replaces: a robust or otherwise non-Gaussian model used to
        // dereference null here.
        gtsam::Matrix2 R_meas = gtsam::Matrix2::Identity() * 0.01;
        if (auto g = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(meas_noise))
            R_meas = g->covariance();

        const gtsam::Matrix2 Rt = pred_pose.rotation().matrix().transpose();
        gtsam::Matrix2 J;
        J << 0.0, 1.0,
            -1.0, 0.0;

        // n rows, m real columns plus one "no match" column per row priced at the
        // gate. The matrix is therefore always at least as wide as it is tall,
        // and gating falls out of the assignment rather than being a separate
        // pass: any pairing costing more than the gate loses to its own column.
        std::vector<std::vector<double>> cost(n, std::vector<double>(m + n, kBlocked));

        for (size_t i = 0; i < n; ++i)
        {
            cost[i][m + i] = mahalanobis_threshold_;
            const gtsam::Point2 local(observations[i].x, observations[i].y);

            // Association is geometry and nothing else. The cone's reported type
            // takes no part even though one is carried: gating on it would spawn
            // a duplicate on top of a real cone whenever a detection disagreed,
            // and a big cone that reads as small for one frame is a misread, not
            // a second cone half a metre away. Types accumulate as votes on the
            // landmark instead, where a misread costs one vote.
            for (size_t j = 0; j < m; ++j)
            {
                const LandmarkRecord &record = landmarks_.at(ids[j]);

                // Reject on plain distance first: the chi-square test costs a 2x2
                // inverse, this costs a subtraction.
                const gtsam::Vector2 innovation = local - pred_pose.transformTo(record.position);
                if (innovation.norm() > max_association_distance_)
                    continue;

                gtsam::Matrix2 P_lm = record.covariance;
                P_lm(0, 0) += covariance_floor_;
                P_lm(1, 1) += covariance_floor_;

                gtsam::Matrix23 H;
                H.leftCols<2>() = -Rt;
                H.rightCols<1>() = Rt * J * (record.position - vehicle);

                const gtsam::Matrix2 S =
                    H * pose_cov_map * H.transpose() + Rt * P_lm * Rt.transpose() + R_meas;

                const double det = S.determinant();
                if (!std::isfinite(det) || det <= 0.0)
                    continue;

                const double d2 = innovation.transpose() * S.inverse() * innovation;
                if (std::isfinite(d2) && d2 < mahalanobis_threshold_)
                    cost[i][j] = d2;
            }
        }

        const std::vector<int> assignment = hungarian(cost);
        for (size_t i = 0; i < n; ++i)
        {
            const int j = assignment[i];
            if (j >= 0 && static_cast<size_t>(j) < m)
                association[i] = ids[j];
        }
        return association;
    }

    void Slam::update_isam()
    {
        using namespace std::chrono;
        const auto start_time = high_resolution_clock::now();

        isam_.update(new_factors_, new_values_);
        current_estimate_ = isam_.calculateEstimate();

        const auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start_time).count();
        this->last_update_elapsed_ms_ = duration / 1.0e3;

        // Extract optimized pose
        gtsam::Symbol curr_x('x', step_count_);
        gtsam::Pose2 optimized_pose = current_estimate_.at<gtsam::Pose2>(curr_x);
        map_to_odom_correction_ = optimized_pose.compose(last_raw_odom_.inverse());

        // One marginal per keyframe -- only the pose -- for the next gate. If it
        // fails, carry the previous uncertainty rather than pretending to be sure.
        try
        {
            pose_cov_ = isam_.marginalCovariance(curr_x);
        }
        catch (const std::exception &)
        {
        }

        // Cleanup
        new_factors_.resize(0);
        new_values_.clear();

        // Refresh landmark positions and regenerate the published map.
        this->last_map_.clear();
        this->last_map_.reserve(landmarks_.size());

        for (auto &[id, record] : landmarks_)
        {
            const gtsam::Symbol l_key('l', id);
            if (current_estimate_.exists(l_key))
                record.position = current_estimate_.at<gtsam::Point2>(l_key);

            // Type and confidence come from accumulated evidence, not from a
            // hardcoded 1.0, so planning can tell a well-observed cone from a
            // contested one.
            this->last_map_.push_back({record.position.x(), record.position.y(),
                                       record.types.best, record.types.confidence(), {}});
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

    int Slam::get_last_matched_count() const
    {
        return last_matched_count_;
    }

    int Slam::get_last_new_count() const
    {
        return last_new_count_;
    }

} // namespace mirena
