#include "slam.hpp"

#include <gtsam/nonlinear/Marginals.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>
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

        /// A usable 2x2 covariance from what perception reported, or the fallback
        /// when it reported nothing.
        ///
        /// Perception publishes zeros when it has no estimate, and a covariance of
        /// zero would mean a perfectly known cone -- the most dangerous thing this
        /// could believe -- so the check is not defensive padding.
        gtsam::Matrix2 measurement_covariance(const std::array<double, 4> &reported,
                                              double fallback_sigma, double min_sigma)
        {
            gtsam::Matrix2 R;
            R << reported[0], reported[1], reported[2], reported[3];

            const bool usable = R.allFinite() && R(0, 0) > 0.0 && R(1, 1) > 0.0 &&
                                R.determinant() > 0.0;
            if (!usable)
                R = gtsam::Matrix2::Identity() * (fallback_sigma * fallback_sigma);

            // Symmetrise, then floor the diagonal. A round trip through a message
            // leaves R(0,1) and R(1,0) very slightly apart, and GTSAM will not
            // accept a noise model that is not symmetric.
            R = (0.5 * (R + R.transpose())).eval();
            const double floor = min_sigma * min_sigma;
            R(0, 0) = std::max(R(0, 0), floor);
            R(1, 1) = std::max(R(1, 1), floor);
            return R;
        }
    } // namespace

    void Slam::TypeVote::add(const std::string &type)
    {
        if (type.empty())
            return;

        const int count = ++counts[type];
        if (count > best_count)
        {
            best_count = count;
            best = type;
        }
    }

    Slam::Slam(const gtsam::Pose2 &initial_raw_pose, SlamParams params)
        : params_(params)
    {
        last_raw_odom_ = initial_raw_pose;
        map_to_odom_ = gtsam::Pose2();

        // The map frame is defined to be the car's starting pose, so the first
        // keyframe sits at the origin and is held there. Everything else is
        // measured from it.
        anchor_pose_ = gtsam::Pose2(0, 0, 0);
        anchor_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e-6);

        Keyframe first;
        first.index = next_keyframe_++;
        first.odom_delta = gtsam::Pose2();
        first.odom_noise = anchor_noise_;
        first.estimate = anchor_pose_;
        window_.push_back(std::move(first));
    }

    void Slam::observe(const gtsam::Pose2 &raw_odom,
                       const std::vector<Observation> &observations,
                       SlamDebug *debug)
    {
        // 1. Where the odometry says we are, measured from the last keyframe.
        const gtsam::Pose2 delta = last_raw_odom_.between(raw_odom);
        const gtsam::Pose2 pred_pose = window_.back().estimate.compose(delta);

        // 2. How much the gate should have widened since that keyframe. There has
        //    been no solve since, so this is guessed forward from distance driven
        //    rather than computed -- which is all the gate needs it for.
        const double driven = delta.translation().norm() + std::abs(delta.theta());
        const double grew = params_.gate_odom_sigma_per_metre * std::max(driven, 1e-3);
        gtsam::Matrix3 growth = gtsam::Matrix3::Identity() * (grew * grew);
        growth(2, 2) *= 0.1;

        const gtsam::Matrix3 adjoint = delta.inverse().AdjointMap();
        const gtsam::Matrix3 pred_cov_body = adjoint * pose_cov_ * adjoint.transpose() + growth;
        const gtsam::Matrix3 A = body_to_map(pred_pose);

        // 3. How well each cone was located, straight from perception.
        std::vector<gtsam::Matrix2> measurement_cov;
        measurement_cov.reserve(observations.size());
        for (const Observation &o : observations)
        {
            measurement_cov.push_back(measurement_covariance(
                o.covariance, params_.default_measurement_sigma, params_.min_measurement_sigma));
        }

        // 4. One landmark per observation, one observation per landmark.
        const std::vector<int> matched =
            associate(pred_pose, A * pred_cov_body * A.transpose(), observations, measurement_cov);

        pending_ = Pending{};
        pending_.raw_odom = raw_odom;
        pending_.estimate = pred_pose;
        pending_.valid = true;

        last_matched_ = 0;
        last_new_ = 0;
        std::vector<int> seen;

        for (size_t i = 0; i < observations.size(); ++i)
        {
            const gtsam::Point2 local(observations[i].x, observations[i].y);
            int id = matched[i];

            if (id < 0)
            {
                // A brand new cone -- unless it lands within touching distance of
                // one already mapped. The assignment is one-to-one, so a second
                // detection of a cone another observation just claimed comes back
                // unmatched, and adding it would stack two landmarks on one
                // physical cone. Cones stand metres apart, so nothing this close
                // can be a new one, and refusing costs a single sighting.
                const gtsam::Point2 world = pred_pose.transformFrom(local);
                const bool crowded = std::any_of(
                    landmarks_.begin(), landmarks_.end(), [&](const auto &entry) {
                        return (entry.second.position - world).norm() < params_.min_cone_separation;
                    });
                if (crowded)
                {
                    if (debug)
                        debug->suppressed.push_back(world);
                    continue;
                }

                id = next_landmark_++;
                LandmarkRecord record;
                record.position = world;
                landmarks_.emplace(id, std::move(record));
                ++last_new_;
                if (debug)
                    debug->created.push_back(world);
            }
            else
            {
                ++last_matched_;
                if (debug)
                    debug->associations.emplace_back(pred_pose.transformFrom(local),
                                                     landmarks_.at(id).position);
            }

            ++landmarks_.at(id).sightings;
            landmarks_.at(id).types.add(observations[i].type);

            pending_.ids.push_back(id);
            pending_.local.push_back(local);
            pending_.noise.push_back(gtsam::noiseModel::Gaussian::Covariance(measurement_cov[i]));
            seen.push_back(id);
        }

        // Judged on every frame the sensor delivers, not on the few that happen
        // to become keyframes.
        remove_unseen(pred_pose, seen, debug);
        publish_map();

        if (debug)
        {
            debug->pose = pred_pose;
            const gtsam::Point2 vehicle = pred_pose.translation();
            for (const auto &[id, record] : landmarks_)
            {
                if ((record.position - vehicle).norm() < params_.local_map_radius)
                    debug->sightings.emplace_back(record.position, record.sightings);
            }
        }
    }

    void Slam::add_keyframe(gtsam::SharedNoiseModel odom_noise, SlamDebug *debug)
    {
        (void)debug;
        if (!pending_.valid)
            return;

        using namespace std::chrono;
        const auto started = high_resolution_clock::now();

        Keyframe frame;
        frame.index = next_keyframe_++;
        frame.odom_delta = last_raw_odom_.between(pending_.raw_odom);
        frame.odom_noise = odom_noise;
        frame.estimate = pending_.estimate;
        frame.ids = pending_.ids;
        frame.local = pending_.local;
        frame.noise = pending_.noise;

        last_raw_odom_ = pending_.raw_odom;
        pending_.valid = false;

        window_.push_back(std::move(frame));
        optimise();
        publish_map();

        map_to_odom_ = window_.back().estimate.compose(last_raw_odom_.inverse());
        last_update_ms_ =
            duration_cast<microseconds>(high_resolution_clock::now() - started).count() / 1.0e3;
    }

    void Slam::publish_map()
    {
        map_.clear();
        map_.reserve(landmarks_.size());
        for (const auto &[id, record] : landmarks_)
            map_.push_back({record.position.x(), record.position.y(), record.types.best});
    }

    void Slam::optimise()
    {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;

        // The window is anchored at its oldest pose. That prior is everything the
        // keyframes already dropped had to say about where the car was, and
        // without it the window would be free to slide and turn as a whole.
        const gtsam::Symbol first('x', window_.front().index);
        graph.addPrior(first, anchor_pose_, anchor_noise_);
        values.insert(first, window_.front().estimate);

        for (size_t k = 1; k < window_.size(); ++k)
        {
            const gtsam::Symbol prev('x', window_[k - 1].index);
            const gtsam::Symbol curr('x', window_[k].index);
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
                prev, curr, window_[k].odom_delta, window_[k].odom_noise);
            values.insert(curr, window_[k].estimate);
        }

        // Every cone the window can see, seeded at its current estimate and held
        // by a prior carrying what earlier keyframes made of it.
        std::unordered_set<int> in_window;
        for (const Keyframe &frame : window_)
            for (const int id : frame.ids)
                in_window.insert(id);

        for (const int id : in_window)
        {
            const LandmarkRecord &record = landmarks_.at(id);
            const gtsam::Symbol key('l', id);
            values.insert(key, record.position);

            // Only cones with sightings older than the window get a prior. One
            // seen entirely from inside it is already pinned by its own
            // measurements, and a prior would be the same evidence twice.
            if (record.has_prior)
            {
                graph.addPrior(key, record.prior_position,
                               gtsam::noiseModel::Gaussian::Covariance(record.prior_covariance));
            }
        }

        // A robust kernel on every measurement, so an association that slipped
        // through the gate is down-weighted rather than believed. Its width is in
        // whitened units, so it scales with each cone's own covariance: a metre of
        // error on a distant cone is not the outlier it would be on one two metres
        // away, and now that perception says which is which, the kernel knows too.
        for (const Keyframe &frame : window_)
        {
            const gtsam::Expression<gtsam::Pose2> x_(gtsam::Symbol('x', frame.index));
            for (size_t i = 0; i < frame.ids.size(); ++i)
            {
                const auto robust = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::DCS::Create(3.0), frame.noise[i]);
                const gtsam::Expression<gtsam::Point2> l_(gtsam::Symbol('l', frame.ids[i]));
                graph.add(gtsam::ExpressionFactor<gtsam::Point2>(
                    robust, frame.local[i], gtsam::transformTo(x_, l_)));
            }
        }

        gtsam::Values result;
        try
        {
            gtsam::LevenbergMarquardtParams lm;
            lm.setMaxIterations(10);
            result = gtsam::LevenbergMarquardtOptimizer(graph, values, lm).optimize();
        }
        catch (const std::exception &)
        {
            // Keep the previous estimate rather than a half-solved one. The next
            // keyframe brings new information and another chance.
            return;
        }

        for (Keyframe &frame : window_)
            frame.estimate = result.at<gtsam::Pose2>(gtsam::Symbol('x', frame.index));
        for (const int id : in_window)
            landmarks_.at(id).position = result.at<gtsam::Point2>(gtsam::Symbol('l', id));

        // Marginals are what make it safe to throw the window away: they are how
        // this solve's confidence survives into the next one. If they cannot be
        // had the estimates still stand, the uncertainties just go stale.
        //
        // Only two things need them, and neither is every landmark every cycle:
        // the newest pose, for the next association gate, and the cones of a
        // keyframe about to be dropped. That is a handful of back-substitutions
        // rather than one per cone in the window.
        try
        {
            gtsam::Marginals marginals(graph, result);
            pose_cov_ = marginals.marginalCovariance(gtsam::Symbol('x', window_.back().index));

            while (static_cast<int>(window_.size()) > params_.window_keyframes)
                freeze_oldest(graph, result);
        }
        catch (const std::exception &)
        {
        }
    }

    void Slam::freeze_oldest(const gtsam::NonlinearFactorGraph &graph,
                             const gtsam::Values &result)
    {
        gtsam::Marginals marginals(graph, result);

        const auto clamp_2d = [](gtsam::Matrix2 cov, double lo, double hi) {
            const double lo2 = lo * lo, hi2 = hi * hi;
            cov(0, 0) = std::min(std::max(cov(0, 0), lo2), hi2);
            cov(1, 1) = std::min(std::max(cov(1, 1), lo2), hi2);
            return cov;
        };

        // Everything the departing keyframe knew about a cone becomes that cone's
        // prior. Frozen here and not touched again until another of its keyframes
        // leaves, so the same measurements are never counted twice.
        for (const int id : window_.front().ids)
        {
            auto it = landmarks_.find(id);
            if (it == landmarks_.end())
                continue;
            LandmarkRecord &record = it->second;
            record.prior_position = record.position;
            record.prior_covariance =
                clamp_2d(marginals.marginalCovariance(gtsam::Symbol('l', id)),
                         params_.min_landmark_sigma, params_.max_landmark_sigma);
            record.covariance = record.prior_covariance;
            record.has_prior = true;
        }

        window_.pop_front();

        // The pose behind it inherits an anchor at its own optimised estimate,
        // with its own uncertainty rather than a guessed one. That is what makes
        // dropping the keyframe safe rather than merely cheap -- and clamping it
        // matters in both directions: too tight and the window cannot correct
        // itself when a cone says the car is elsewhere, too loose and it drifts.
        anchor_pose_ = window_.front().estimate;
        gtsam::Matrix3 cov = marginals.marginalCovariance(
            gtsam::Symbol('x', window_.front().index));
        const double lo = params_.min_anchor_sigma * params_.min_anchor_sigma;
        const double hi = params_.max_anchor_sigma * params_.max_anchor_sigma;
        for (int k = 0; k < 3; ++k)
            cov(k, k) = std::min(std::max(cov(k, k), lo), hi);
        anchor_noise_ = gtsam::noiseModel::Gaussian::Covariance(cov);
    }


    void Slam::remove_unseen(const gtsam::Pose2 &pose,
                             const std::vector<int> &seen,
                             SlamDebug *debug)
    {
        last_removed_ = 0;
        const std::unordered_set<int> observed(seen.begin(), seen.end());

        for (auto it = landmarks_.begin(); it != landmarks_.end();)
        {
            LandmarkRecord &record = it->second;
            if (observed.count(it->first))
            {
                ++it;
                continue;
            }

            // Was it somewhere the car could actually have seen it? A cone behind
            // the car has not gone missing -- nothing was pointed at it.
            const gtsam::Point2 local = pose.transformTo(record.position);
            const bool in_view =
                local.norm() < params_.view_range && local.x() > 0.0 &&
                std::abs(std::atan2(local.y(), local.x())) < params_.view_half_angle;
            if (!in_view)
            {
                ++it;
                continue;
            }

            // Being seen buys patience, and it buys a bounded amount of it.
            // Misses are never wiped by a later sighting -- a cone that is seen
            // once in a while and missed the rest of the time is not a cone, and
            // resetting the count on every sighting is what let those live for
            // ever.
            const int allowance =
                params_.max_misses * std::min(record.sightings, params_.confirm_sightings);
            if (++record.misses < allowance)
            {
                ++it;
                continue;
            }

            // Looked at repeatedly, never seen again: it was never a cone.
            if (debug)
                debug->removed.push_back(record.position);
            const int id = it->first;
            it = landmarks_.erase(it);
            ++last_removed_;

            // Its measurements go with it, or the next graph would reference a
            // landmark that no longer exists.
            for (Keyframe &frame : window_)
            {
                for (size_t i = frame.ids.size(); i-- > 0;)
                {
                    if (frame.ids[i] != id)
                        continue;
                    frame.ids.erase(frame.ids.begin() + static_cast<long>(i));
                    frame.local.erase(frame.local.begin() + static_cast<long>(i));
                    frame.noise.erase(frame.noise.begin() + static_cast<long>(i));
                }
            }
        }
    }

    std::vector<int> Slam::associate(const gtsam::Pose2 &pred_pose,
                                     const gtsam::Matrix3 &pose_cov_map,
                                     const std::vector<Observation> &observations,
                                     const std::vector<gtsam::Matrix2> &measurement_cov) const
    {
        const size_t n = observations.size();
        std::vector<int> association(n, -1);
        if (n == 0)
            return association;

        // Only what is plausibly in view.
        std::vector<int> ids;
        const gtsam::Point2 vehicle = pred_pose.translation();
        for (const auto &[id, record] : landmarks_)
        {
            if ((record.position - vehicle).norm() < params_.local_map_radius)
                ids.push_back(id);
        }
        if (ids.empty())
            return association;

        const size_t m = ids.size();
        const gtsam::Matrix2 Rt = pred_pose.rotation().matrix().transpose();
        gtsam::Matrix2 J;
        J << 0.0, 1.0,
            -1.0, 0.0;

        // n rows, m real columns plus one "no match" column per row priced at the
        // gate. The matrix is therefore always at least as wide as it is tall, and
        // gating falls out of the assignment rather than being a separate pass:
        // any pairing costing more than the gate loses to its own column.
        std::vector<std::vector<double>> cost(n, std::vector<double>(m + n, kBlocked));

        for (size_t i = 0; i < n; ++i)
        {
            cost[i][m + i] = params_.mahalanobis_threshold;
            const gtsam::Point2 local(observations[i].x, observations[i].y);

            // Association is geometry and nothing else. The cone's reported type
            // takes no part even though one is carried: gating on it would spawn a
            // duplicate on top of a real cone whenever a detection disagreed, and
            // a big cone that reads as small for one frame is a misread, not a
            // second cone half a metre away. Types accumulate as votes on the
            // landmark instead, where a misread costs one vote.
            for (size_t j = 0; j < m; ++j)
            {
                const LandmarkRecord &record = landmarks_.at(ids[j]);

                // Reject on plain distance first: the chi-square test costs a 2x2
                // inverse, this costs a subtraction.
                const gtsam::Vector2 innovation = local - pred_pose.transformTo(record.position);
                if (innovation.norm() > params_.max_association_distance)
                    continue;

                gtsam::Matrix2 P_lm = record.covariance;
                P_lm(0, 0) += params_.covariance_floor;
                P_lm(1, 1) += params_.covariance_floor;

                gtsam::Matrix23 H;
                H.leftCols<2>() = -Rt;
                H.rightCols<1>() = Rt * J * (record.position - vehicle);

                const gtsam::Matrix2 S = H * pose_cov_map * H.transpose() +
                                         Rt * P_lm * Rt.transpose() + measurement_cov[i];

                const double det = S.determinant();
                if (!std::isfinite(det) || det <= 0.0)
                    continue;

                const double d2 = innovation.transpose() * S.inverse() * innovation;
                if (std::isfinite(d2) && d2 < params_.mahalanobis_threshold)
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

} // namespace mirena
