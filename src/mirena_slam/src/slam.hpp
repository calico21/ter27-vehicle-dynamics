#pragma once

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/expressions.h>

#include <array>
#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mirena
{

    /// One cone as perception saw it, in the car frame.
    struct Observation
    {
        double x, y;
        /// What perception could tell about the cone without reading its colour,
        /// which in a colourless pipeline is its size: `cone_big_orange` for the
        /// cones marking the start line, `unknown_cone` for the rest.
        std::string type;
        /// How well perception located it: 2x2 row-major [xx, xy, yx, yy], in the
        /// car frame. All zeros means it had no estimate and the default is used.
        std::array<double, 4> covariance{};
    };

    /// One cone in the map.
    struct Landmark
    {
        double x, y;
        std::string type;
    };

    /// What one keyframe did, for the debug topic. Filled only when a debug sink
    /// is handed in, so watching costs nothing when nobody is.
    ///
    /// Between them these answer the question the map alone cannot: not "where
    /// are the cones" but "why is the map the shape it is". Associations with
    /// long innovations mean a pose drifting; a steady drip of `created` after
    /// the first lap means association failing to recognise cones already mapped;
    /// `suppressed` filling up means the gate is too tight.
    struct SlamDebug
    {
        gtsam::Pose2 pose;
        /// Observation (in map frame) paired with the landmark it matched. The
        /// length of each pair is the innovation the gate accepted.
        std::vector<std::pair<gtsam::Point2, gtsam::Point2>> associations;
        std::vector<gtsam::Point2> created;
        /// Detections dropped for landing on top of a cone already mapped.
        std::vector<gtsam::Point2> suppressed;
        /// Landmarks deleted for never being seen again.
        std::vector<gtsam::Point2> removed;
        /// Landmarks near the car, with how many times each has been seen.
        std::vector<std::pair<gtsam::Point2, int>> sightings;
    };

    struct SlamParams
    {
        // -- What the optimiser is given ----------------------------------------

        /// How many keyframes are optimised at once.
        ///
        /// This is what bounds the whole system. Everything older is folded into
        /// priors and dropped, so the problem solved on lap ten is the same size
        /// as the one solved on lap one, and takes the same time. It has to be
        /// long enough that a cone stays in the window from the keyframe that
        /// first sees it to the one that last does.
        int window_keyframes = 25;

        /// How fast the pose gate widens between keyframes, per metre driven.
        /// Only used for association: between keyframes there is no solve, so the
        /// covariance has to be guessed forward rather than computed.
        double gate_odom_sigma_per_metre = 0.05;

        /// Measurement sigma used when perception reports no covariance, metres.
        double default_measurement_sigma = 0.10;
        /// Floor on the measurement sigma, metres. Stops a confidently reported
        /// detection being trusted further than the extrinsics deserve.
        double min_measurement_sigma = 0.03;

        // -- Data association ---------------------------------------------------

        /// Chi-square gate on the squared Mahalanobis distance. 5.99 is the 95th
        /// percentile for two degrees of freedom, 9.21 the 99th.
        double mahalanobis_threshold = 5.99;
        /// Hard Euclidean backstop, metres, applied before the chi-square test.
        /// Sized so it does not bind in normal operation; it exists to stop a
        /// pose that has gone badly wrong from reaching across the track.
        double max_association_distance = 2.5;
        /// Nothing new is created within this of a cone already mapped. Cones
        /// stand metres apart, so anything closer is a second report of one the
        /// frame already has.
        double min_cone_separation = 1.0;
        /// Added to a landmark's covariance before gating, in m^2. A converged
        /// cone otherwise carries a gate a couple of centimetres wide, and every
        /// re-detection that misses it becomes a duplicate sitting on top of it.
        double covariance_floor = 0.09;
        /// Loosest and tightest a frozen landmark's prior may be, in metres.
        ///
        /// The floor is not numerical hygiene. A cone frozen at 3 cm cannot move,
        /// so when the car comes back round and its own drift says the cone is
        /// somewhere else, the map wins and the pose is dragged to fit a map that
        /// was itself built from a drifting pose. Leaving the cone room to be
        /// wrong is what lets the loop close.
        double min_landmark_sigma = 0.15;
        double max_landmark_sigma = 2.0;
        /// Loosest and tightest the anchor on the oldest kept pose may be.
        double min_anchor_sigma = 0.05;
        double max_anchor_sigma = 1.0;
        /// Only cones this close to the car are considered for association.
        double local_map_radius = 20.0;

        // -- Removing landmarks that were never there ---------------------------

        /// How many keyframes a cone may be looked straight at and not seen
        /// before it is deleted, if it has only ever been seen once.
        ///
        /// "Looked straight at" carries the weight: a cone behind the car, or off
        /// to the side of a forward-facing sensor, is not missing -- nothing is
        /// pointed at it. Only a cone the car should have seen and did not counts
        /// against it.
        int max_misses = 8;
        /// How much more forgiving each confirmed sighting makes it.
        ///
        /// The allowance is `max_misses * min(sightings, confirm_sightings)`, so a
        /// cone seen once is cheap to drop and one seen repeatedly is expensive --
        /// but nothing is ever immune. That last part is the point: a landmark
        /// that stops being seen has stopped being there, however convincing it
        /// looked on the first lap.
        int confirm_sightings = 3;
        /// Range within which a cone should have been seen, metres.
        double view_range = 12.0;
        /// Half-angle of the sensor's field of view, radians.
        double view_half_angle = 1.05;
    };

    /// Landmark SLAM over a sliding window of keyframes.
    ///
    /// The window is the whole design. Only the last `window_keyframes` poses are
    /// variables; everything before them has been folded into a prior on the
    /// oldest one still kept, and into a prior on each landmark. So the problem
    /// handed to the optimiser is the same size on lap ten as on lap one, which
    /// is what makes memory and time bounded on a track the car goes round twenty
    /// times, and what makes it safe to throw old keyframes away at all.
    ///
    /// A cone gets a prior only once a keyframe that saw it has been dropped,
    /// and that prior is frozen at the moment of dropping rather than refreshed
    /// afterwards. Both halves matter. A cone whose every sighting is still in
    /// the window is already fully described by those measurements, so a prior on
    /// top of it counts the same evidence twice; and refreshing a prior from a
    /// solve that contained it is a feedback loop that drives its covariance to
    /// zero, pinning the map to wherever each cone happened to be first seen and
    /// then dragging the pose to fit it.
    ///
    /// What is still approximate: the frozen priors treat cones as independent,
    /// where a real marginalisation would carry the cross-covariances between
    /// them. That is a much bigger and far less readable piece of machinery, and
    /// this is a cone field.
    class Slam
    {
    public:
        explicit Slam(const gtsam::Pose2 &initial_raw_pose, SlamParams params = {});

        /// Matches one frame of cones against the map, promotes what is new, and
        /// drops what was never there. Call on every frame perception delivers.
        ///
        /// Deliberately separate from add_keyframe, and that separation is the
        /// point. Whether a blob is a cone is decided by looking at it repeatedly,
        /// and detections arrive several times faster than the car moves far
        /// enough to be worth a new pose. Tying the two together threw away most
        /// of the evidence: a phantom got judged on the four or five frames that
        /// happened to coincide with a keyframe rather than on the forty the
        /// sensor actually gave.
        ///
        /// Touches no factors. The graph only ever learns about cones that were
        /// still alive when a keyframe came round.
        ///
        /// @param raw_odom absolute raw odometry for this frame
        void observe(const gtsam::Pose2 &raw_odom,
                     const std::vector<Observation> &observations,
                     SlamDebug *debug = nullptr);

        /// Commits the cones from the most recent observe() as a keyframe and
        /// re-solves. Call when the car has moved far enough to be worth a pose.
        void add_keyframe(gtsam::SharedNoiseModel odom_noise, SlamDebug *debug = nullptr);

        const std::vector<Landmark> &get_map() const { return map_; }
        gtsam::Pose2 get_correction() const { return map_to_odom_; }
        double get_last_update_elapsed_ms() const { return last_update_ms_; }
        int get_last_matched_count() const { return last_matched_; }
        int get_last_new_count() const { return last_new_; }
        int get_last_removed_count() const { return last_removed_; }
        /// Poses currently being optimised. Constant once the window is full,
        /// which is the point of having one.
        int get_window_size() const { return static_cast<int>(window_.size()); }

    private:
        /// One pose in the window, and what was seen from it.
        struct Keyframe
        {
            int index;               ///< index of its 'x' symbol
            gtsam::Pose2 odom_delta; ///< from the previous keyframe
            gtsam::SharedNoiseModel odom_noise;
            gtsam::Pose2 estimate;
            /// Landmark id, where it was seen in the car frame, and how well.
            std::vector<int> ids;
            std::vector<gtsam::Point2> local;
            std::vector<gtsam::SharedNoiseModel> noise;
        };

        /// Which type has been reported for one landmark most often.
        ///
        /// A plain tally, because a colourless detector has nothing graded to
        /// say -- a cluster is tall enough to be a big cone or it is not. Voting
        /// rather than latching means a single misread costs one vote, which
        /// matters because the big cones are what the planner roots the track on.
        struct TypeVote
        {
            std::map<std::string, int> counts;
            std::string best;
            int best_count = 0;

            void add(const std::string &type);
        };

        struct LandmarkRecord
        {
            gtsam::Point2 position;
            /// Everything this cone was worth to the keyframes that have already
            /// left the window, frozen at the moment the last of them left.
            ///
            /// Only landmarks with `has_prior` get one. A cone whose every
            /// sighting is still inside the window needs no prior -- those
            /// measurements are in the graph, and adding a prior on top of them
            /// would be counting the same evidence twice. Getting that wrong is
            /// what pinned the map to wherever each cone was first seen.
            gtsam::Point2 prior_position;
            gtsam::Matrix2 prior_covariance = gtsam::Matrix2::Identity() * 0.25;
            bool has_prior = false;
            /// Uncertainty used for gating. Same as the prior once frozen, and a
            /// standing guess before that.
            gtsam::Matrix2 covariance = gtsam::Matrix2::Identity() * 0.25;
            int sightings = 0;
            /// Keyframes where this cone was in plain view and was not seen.
            int misses = 0;
            TypeVote types;
        };

        /// Nearest-neighbour association, scored by Mahalanobis distance and
        /// solved as an optimal one-to-one assignment.
        ///
        /// The innovation covariance is
        ///
        ///   S = H_x P_pose H_x^T + R^T P_lm R + R_meas,
        ///   H_x = [ -R^T | R^T J (l - p) ],  J = [[0, 1], [-1, 0]]
        ///
        /// so the gate widens exactly where the estimate is genuinely uncertain.
        /// The heading column is not a refinement: at 10 m a yaw sigma of
        /// 0.05 rad is half a metre of lateral error, which dominates everything
        /// else in S, and dropping it leaves the gate far too tight at range.
        ///
        /// The assignment is the other half. Scoring each observation
        /// independently lets two cones detected in one frame both claim a single
        /// landmark, which puts two contradictory measurement factors on one point
        /// and drags it to the midpoint of two real cones.
        ///
        /// @return association[i] is the landmark id matched to observations[i],
        ///         or -1 for none.
        std::vector<int> associate(const gtsam::Pose2 &pred_pose,
                                   const gtsam::Matrix3 &pose_cov_map,
                                   const std::vector<Observation> &observations,
                                   const std::vector<gtsam::Matrix2> &measurement_cov) const;

        /// Solves the window and writes the answer back into the keyframes and
        /// the landmarks.
        void optimise();

        /// Rebuilds the published map from the landmarks.
        void publish_map();

        /// Counts a miss against every landmark that was in plain view and not
        /// seen, and deletes the ones that have run out of chances.
        void remove_unseen(const gtsam::Pose2 &pose,
                           const std::vector<int> &seen,
                           SlamDebug *debug);

        /// Freezes what the oldest keyframe knew into priors and drops it.
        /// Called from inside optimise(), where the solved graph is still in
        /// scope, because that is the only place the marginals can be had.
        void freeze_oldest(const gtsam::NonlinearFactorGraph &graph,
                           const gtsam::Values &result);

        /// The last frame observe() matched, waiting to become a keyframe. Only
        /// the newest is kept: the frames in between have already done their job
        /// of confirming and killing candidates, and adding a pose for each would
        /// give the optimiser a pile of near-identical poses to chew on.
        struct Pending
        {
            gtsam::Pose2 raw_odom;
            gtsam::Pose2 estimate;
            std::vector<int> ids;
            std::vector<gtsam::Point2> local;
            std::vector<gtsam::SharedNoiseModel> noise;
            bool valid = false;
        };

        SlamParams params_;
        Pending pending_;

        std::deque<Keyframe> window_;
        std::map<int, LandmarkRecord> landmarks_;

        /// Prior on the oldest pose in the window: everything the keyframes that
        /// have been dropped had to say about where the car was.
        gtsam::Pose2 anchor_pose_;
        gtsam::SharedNoiseModel anchor_noise_;

        gtsam::Pose2 last_raw_odom_;
        gtsam::Pose2 map_to_odom_;
        /// Covariance of the newest pose, in the map frame, for the next gate.
        gtsam::Matrix3 pose_cov_ = gtsam::Matrix3::Identity() * 1e-9;

        std::vector<Landmark> map_;
        int next_keyframe_ = 0;
        int next_landmark_ = 0;
        double last_update_ms_ = 0.0;
        int last_matched_ = 0;
        int last_new_ = 0;
        int last_removed_ = 0;
    };

} // namespace mirena
