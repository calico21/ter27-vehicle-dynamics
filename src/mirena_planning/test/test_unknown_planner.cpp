#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mirena_planning/unknown_planner.hpp"

using namespace mirena::planning;  // NOLINT(build/namespaces)

namespace
{

constexpr double kRadius = 15.0;
constexpr double kWidth = 3.0;

/// A closed circuit driven anticlockwise from the map origin.
///
/// The map frame is the pose the car started from, so the start gate sits at
/// the origin with the car facing +x. Blue is on the driver's left, which on an
/// anticlockwise lap is the inside of the circle.
std::vector<Observation> ovalTrack(double from_angle = 0.0, double to_angle = 2.0 * M_PI)
{
  const Vec2 centre{0.0, kRadius};
  std::vector<Observation> cones{
    {{0.0, kWidth / 2.0}, Cone::BigOrange, 1.0},
    {{0.0, -kWidth / 2.0}, Cone::BigOrange, 1.0}};

  const double step = 3.0 / kRadius;
  for (double a = from_angle + step; a < to_angle; a += step) {
    const Vec2 radial{std::sin(a), -std::cos(a)};
    cones.push_back({centre + radial * (kRadius - kWidth / 2.0), Cone::Blue, 1.0});
    cones.push_back({centre + radial * (kRadius + kWidth / 2.0), Cone::Yellow, 1.0});
  }
  return cones;
}

/// A straight corridor with its own big orange gate at the far end, the layout
/// that used to drag the start gate into the middle of the track.
std::vector<Observation> straightTrack()
{
  std::vector<Observation> cones{
    {{0.0, kWidth / 2.0}, Cone::BigOrange, 1.0},
    {{0.0, -kWidth / 2.0}, Cone::BigOrange, 1.0},
    {{60.0, kWidth / 2.0}, Cone::BigOrange, 1.0},
    {{60.0, -kWidth / 2.0}, Cone::BigOrange, 1.0}};

  for (double x = 3.0; x <= 57.0; x += 3.0) {
    cones.push_back({{x, kWidth / 2.0}, Cone::Blue, 1.0});
    cones.push_back({{x, -kWidth / 2.0}, Cone::Yellow, 1.0});
  }
  return cones;
}

}  // namespace

TEST(UnknownPlanner, TracesTheCentrelineOfAClosedCircuit)
{
  UnknownPlanner planner;
  const auto plan = planner.plan(ovalTrack(), nullptr);
  ASSERT_TRUE(plan.has_value());
  EXPECT_GT(plan->gates.size(), 25u);

  const Vec2 centre{0.0, kRadius};
  for (const Gate & gate : plan->gates) {
    // Every gate should sit on the centreline circle, to within the sampling.
    EXPECT_NEAR(distance(gate.p, centre), kRadius, 0.35);
  }
}

TEST(UnknownPlanner, OrdersGatesWithoutJumpingAcrossTheTrack)
{
  UnknownPlanner planner;
  const auto plan = planner.plan(ovalTrack(), nullptr);
  ASSERT_TRUE(plan.has_value());

  const Vec2 centre{0.0, kRadius};
  const Vec2 first = plan->gates.front().p - centre;
  double previous = std::atan2(first.y, first.x);
  double travelled = 0.0;
  for (std::size_t i = 1; i < plan->gates.size(); ++i) {
    EXPECT_LT(distance(plan->gates[i].p, plan->gates[i - 1].p), 8.0);

    const Vec2 radial = plan->gates[i].p - centre;
    const double angle = std::atan2(radial.y, radial.x);
    const double delta = wrapAngle(angle - previous);
    // A cross-track jump or a doubling back shows up here as a step that is
    // not a small advance in the same direction.
    EXPECT_GT(delta, 0.0);
    EXPECT_LT(delta, 0.6);
    travelled += delta;
    previous = angle;
  }
  EXPECT_GT(travelled, 2.0 * M_PI - 0.6);
}

TEST(UnknownPlanner, HoldsTheClosedFlagUntilTheDetectionIsStable)
{
  UnknownPlannerParams params;
  params.closure_hysteresis = 5;
  UnknownPlanner planner(params);

  const auto cones = ovalTrack();
  for (int i = 0; i < 4; ++i) {
    const auto plan = planner.plan(cones, nullptr);
    ASSERT_TRUE(plan.has_value());
    EXPECT_FALSE(plan->closed) << "flipped after only " << i + 1 << " agreeing cycles";
  }
  const auto settled = planner.plan(cones, nullptr);
  ASSERT_TRUE(settled.has_value());
  EXPECT_TRUE(settled->closed);
  EXPECT_NEAR(distance(settled->start, settled->finish), 0.0, 1e-9);
}

TEST(UnknownPlanner, LeavesAPartialLapOpen)
{
  UnknownPlanner planner;
  const auto cones = ovalTrack(0.0, M_PI);
  for (int i = 0; i < 10; ++i) {
    const auto plan = planner.plan(cones, nullptr);
    ASSERT_TRUE(plan.has_value());
    EXPECT_FALSE(plan->closed);
  }
}

TEST(UnknownPlanner, TakesTheStartGateFromTheLineNotTheAverageOrangeCone)
{
  UnknownPlanner planner;
  const auto plan = planner.plan(straightTrack(), nullptr);
  ASSERT_TRUE(plan.has_value());

  // The far gate is big orange too; averaging both would put the start at
  // x = 30 and send the trace off from the middle of the track.
  EXPECT_NEAR(plan->start.x, 0.0, 0.5);
  EXPECT_NEAR(plan->start.y, 0.0, 0.5);
  EXPECT_FALSE(plan->closed);
  EXPECT_GT(plan->gates.back().p.x, 50.0);
}

TEST(UnknownPlanner, HeadsForwardFromTheStartLine)
{
  UnknownPlanner planner;
  const auto plan = planner.plan(straightTrack(), nullptr);
  ASSERT_TRUE(plan.has_value());
  ASSERT_GE(plan->gates.size(), 2u);
  EXPECT_GT(plan->gates[1].p.x, plan->gates[0].p.x);
  EXPECT_NEAR(plan->gates[1].psi, 0.0, 0.2);
}

TEST(UnknownPlanner, DeclinesToPlanWithoutAStartGate)
{
  UnknownPlanner planner;
  std::vector<Observation> cones;
  for (double x = 3.0; x <= 30.0; x += 3.0) {
    cones.push_back({{x, kWidth / 2.0}, Cone::Blue, 1.0});
    cones.push_back({{x, -kWidth / 2.0}, Cone::Yellow, 1.0});
  }
  EXPECT_FALSE(planner.plan(cones, nullptr).has_value());
}

TEST(UnknownPlanner, DeclinesToPlanWithoutBoundaries)
{
  UnknownPlanner planner;
  const std::vector<Observation> cones{
    {{0.0, 1.5}, Cone::BigOrange, 1.0},
    {{0.0, -1.5}, Cone::BigOrange, 1.0}};
  EXPECT_FALSE(planner.plan(cones, nullptr).has_value());
}

TEST(UnknownPlanner, FillsDebugGeometryOnlyWhenAsked)
{
  UnknownPlanner planner;
  const auto cones = ovalTrack();

  PlanDebug quiet;
  ASSERT_TRUE(planner.plan(cones, &quiet).has_value());
  EXPECT_TRUE(quiet.edges.empty());
  EXPECT_TRUE(quiet.midpoints.empty());
  EXPECT_TRUE(quiet.has_start);

  PlanDebug verbose;
  verbose.verbose = true;
  ASSERT_TRUE(planner.plan(cones, &verbose).has_value());
  EXPECT_FALSE(verbose.edges.empty());
  EXPECT_FALSE(verbose.gate_edges.empty());
  EXPECT_FALSE(verbose.midpoints.empty());
  EXPECT_LT(verbose.gate_edges.size(), verbose.edges.size());
}

TEST(UnknownPlanner, ResetForgetsTheClosedTrack)
{
  UnknownPlanner planner;
  const auto cones = ovalTrack();
  for (int i = 0; i < 6; ++i) {planner.plan(cones, nullptr);}
  ASSERT_TRUE(planner.closed());

  planner.reset();
  EXPECT_FALSE(planner.closed());
  const auto plan = planner.plan(cones, nullptr);
  ASSERT_TRUE(plan.has_value());
  EXPECT_FALSE(plan->closed);
}
