// Whole runs over real track layouts.
//
// The layouts come out of the sim's Export Track button and are the same ones
// mirena_planning tests against. A fixture is a shape somebody thought of; these
// are whatever the generator produced, which is the point.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "track_sim.hpp"

using namespace mirena;       // NOLINT(build/namespaces)
using namespace mirena::test; // NOLINT(build/namespaces)

namespace
{

    std::vector<std::string> track_names()
    {
        std::vector<std::string> names;
        for (int i = 1; i <= 10; ++i)
            names.push_back("track" + std::to_string(i));
        return names;
    }

    /// Closest pair among a set of points, or infinity if there are fewer than two.
    double closest_pair(const std::vector<gtsam::Point2> &points)
    {
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < points.size(); ++i)
            for (size_t j = i + 1; j < points.size(); ++j)
                best = std::min(best, (points[i] - points[j]).norm());
        return best;
    }

    int neighbours_within(const std::vector<Landmark> &map, double limit)
    {
        int count = 0;
        for (size_t i = 0; i < map.size(); ++i)
        {
            const gtsam::Point2 a(map[i].x, map[i].y);
            for (size_t j = 0; j < map.size(); ++j)
            {
                if (i == j)
                    continue;
                if ((gtsam::Point2(map[j].x, map[j].y) - a).norm() < limit)
                {
                    ++count;
                    break;
                }
            }
        }
        return count;
    }

} // namespace

class TrackScenario : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TrackScenario, MapsTheTrackWithoutDuplicating)
{
    const Track track = load_track(GetParam());
    SimParams params;
    params.seed = 42;
    Simulator sim(track, params);

    Slam slam(track.start);
    const RunResult result = drive(slam, sim, 5);

    ASSERT_TRUE(result.healthy) << GetParam() << ": solver gave up";
    ASSERT_FALSE(result.map.empty()) << GetParam() << ": mapped nothing at all";

    // Every mapped cone should be near a real one. This is the assertion that
    // catches phantoms surviving into the graph.
    int spurious = 0;
    for (const Landmark &l : result.map)
    {
        double best = std::numeric_limits<double>::infinity();
        for (const Cone &cone : track.cones)
            best = std::min(best, (cone.position - gtsam::Point2(l.x, l.y)).norm());
        if (best > 1.0)
            ++spurious;
    }

    // And most real cones the car drove past should have been found.
    int found = 0;
    for (const Cone &cone : track.cones)
        if (nearest(result.map, cone.position) < 1.0)
            ++found;

    const double coverage =
        static_cast<double>(found) / static_cast<double>(track.cones.size());
    std::printf("  %-9s cones %3zu  mapped %3zu  coverage %4.0f%%  spurious %d  "
                "keyframes %d\n",
                GetParam().c_str(), track.cones.size(), result.map.size(),
                100.0 * coverage, spurious, result.keyframes);

    EXPECT_GT(coverage, 0.80) << GetParam() << ": too much of the track is missing";
    EXPECT_LE(spurious, 3) << GetParam() << ": landmarks that match no real cone";
    EXPECT_LE(neighbours_within(result.map, 0.9), 4)
        << GetParam() << ": duplicates sitting on top of each other";
}

INSTANTIATE_TEST_SUITE_P(RealTracks, TrackScenario,
                         ::testing::ValuesIn(track_names()),
                         [](const ::testing::TestParamInfo<std::string> &info) {
                             return info.param;
                         });

TEST(Scenario, ClosePairIsNotCollapsed)
{
    // The negative obligation. Getting the start-line cones detected and in range
    // is perception's job; not destroying a pair it did resolve is this class's.
    //
    // Parameterised over approach angle because head-on is the hard one: it puts
    // the separation across the beam, where the sensor is at its worst, and it is
    // also the nominal approach.
    for (const double degrees : {0.0, 15.0, 30.0, 45.0})
    {
        const double angle = degrees * M_PI / 180.0;
        Track track = load_track("track1");

        // A pair 0.9 m apart, straddling the path a little way ahead.
        const gtsam::Point2 centre(14.0, 0.0);
        const gtsam::Point2 offset(0.45 * std::sin(angle), 0.45 * std::cos(angle));
        track.cones.push_back({centre + offset, kBig});
        track.cones.push_back({centre - offset, kBig});

        SimParams params;
        params.seed = static_cast<unsigned>(101 + degrees);
        Simulator sim(track, params);

        Slam slam(track.start);
        const RunResult result = drive(slam, sim, 3);
        ASSERT_TRUE(result.healthy);

        int near_pair = 0;
        for (const Landmark &l : result.map)
            if ((gtsam::Point2(l.x, l.y) - centre).norm() < 2.0)
                ++near_pair;

        EXPECT_EQ(near_pair, 2)
            << "approach " << degrees << " deg: the pair became " << near_pair
            << " landmark(s)";
    }
}

TEST(Scenario, BigConesKeepTheirType)
{
    // Types are votes and nothing branches on them, but they are published and
    // the planner roots the track on them, so a big cone must not come out small.
    Track track = load_track("track1");
    SimParams params;
    params.seed = 77;
    Simulator sim(track, params);

    Slam slam(track.start);
    const RunResult result = drive(slam, sim, 3);
    ASSERT_TRUE(result.healthy);

    const std::vector<gtsam::Point2> big_truth = [&] {
        std::vector<gtsam::Point2> out;
        for (const Cone &c : track.cones)
            if (c.type == kBig)
                out.push_back(c.position);
        return out;
    }();
    ASSERT_FALSE(big_truth.empty()) << "this track has no big cones to check";

    const std::vector<gtsam::Point2> big_mapped = of_type(result.map, kBig);
    std::printf("  big cones: %zu on the track, %zu in the map\n", big_truth.size(),
                big_mapped.size());

    for (const gtsam::Point2 &truth : big_truth)
    {
        double best = std::numeric_limits<double>::infinity();
        for (const gtsam::Point2 &m : big_mapped)
            best = std::min(best, (m - truth).norm());
        EXPECT_LT(best, 1.5) << "a big cone was not mapped as big";
    }
}

TEST(Scenario, NoDuplicatesAfterTheFirstLap)
{
    // Once the track has been driven, association should recognise what it
    // already has. A steady drip of promotions after that is the symptom the
    // whole gate exists to prevent.
    Track track = load_track("track2");
    SimParams params;
    params.seed = 13;
    Simulator sim(track, params);

    Slam slam(track.start);
    gtsam::Pose2 last_keyframe_raw = track.start;
    bool first = true;
    int promoted_first_lap = 0, promoted_later = 0;

    Frame frame;
    while (sim.step(frame, 5))
    {
        const gtsam::Vector3 sigmas = sim.odom_sigmas(last_keyframe_raw);
        slam.observe(frame.raw_odom, frame.observations, sigmas);
        if (first || slam.wants_keyframe())
        {
            const int before = slam.get_landmark_count();
            slam.add_keyframe(sigmas);
            last_keyframe_raw = frame.raw_odom;
            first = false;
            const int gained = slam.get_landmark_count() - before;
            (sim.laps_done() < 1 ? promoted_first_lap : promoted_later) += gained;
        }
    }

    std::printf("  promoted: %d on the first lap, %d over the next four\n",
                promoted_first_lap, promoted_later);

    ASSERT_GT(promoted_first_lap, 10) << "did not map the track on the first lap";
    EXPECT_LT(promoted_later, promoted_first_lap / 2)
        << "cones keep being discovered on ground already driven";
}

TEST(Scenario, CorrectionIsContinuous)
{
    // map->odom is what the MPC projects the car onto, and its spline search is a
    // local one. A large step at loop closure mis-projects the arc length and
    // breaks lap counting. The node filters this with a half-second time
    // constant; this measures what the filter is being asked to absorb.
    Track track = load_track("track1");
    SimParams params;
    params.seed = 31;
    Simulator sim(track, params);

    Slam slam(track.start);
    const RunResult result = drive(slam, sim, 5);
    ASSERT_TRUE(result.healthy);
    ASSERT_FALSE(result.correction_jump_xy.empty());

    std::vector<double> jumps = result.correction_jump_xy;
    std::sort(jumps.begin(), jumps.end());
    const double p50 = jumps[jumps.size() / 2];
    const double worst = jumps.back();
    const double worst_theta = *std::max_element(result.correction_jump_theta.begin(),
                                                 result.correction_jump_theta.end());

    std::printf("  correction step: p50 %.3f m   max %.3f m   max heading %.3f rad\n",
                p50, worst, worst_theta);

    EXPECT_LT(p50, 0.05) << "the correction is noisy every keyframe, not just at closure";
    EXPECT_LT(worst, 1.5) << "a correction step this large will mis-project the MPC";
}
