// Centreline planner for missions where the track is discovered on the fly
// (trackdrive, autocross).

#pragma once

#include <optional>
#include <vector>

#include "mirena_planning/planner.hpp"

namespace mirena::planning
{

struct UnknownPlannerParams
{
  /// Widest blue<->yellow Delaunay edge still plausible as a track gate. Longer
  /// edges are the diagonals that cut across hairpins.
  double max_gate_width{4.5};
  /// Longest step the centreline trace may take between consecutive gates.
  double max_gate_gap{8.0};
  /// How strongly the trace prefers heading continuity over sheer proximity.
  double turn_penalty{1.5};
  /// Minimum cosine between the current heading and the next step: rejects
  /// turns sharper than ~84 degrees, and with them every backwards jump.
  double min_progress_cos{0.1};
  /// Distance to the start gate that counts as "back at the line".
  double closure_radius{8.0};
  /// Near-coincident midpoints from adjacent triangles merge within this.
  double merge_radius{0.10};
  /// Radius around the map origin searched for the start gate. The map frame is
  /// anchored at the car's start pose, so the start line is at the origin; this
  /// is what stops an autocross finish gate from being averaged in with it.
  double start_gate_radius{10.0};
  /// Cones below this confidence are ignored.
  double min_confidence{0.0};
  /// A path shorter than this cannot be a closed lap.
  int min_path_points{15};
  /// Consecutive agreeing detections needed to flip the published closed flag.
  /// Without it the flag chatters and the controller keeps re-fitting its
  /// spline between periodic and open.
  int closure_hysteresis{5};
};

/// Delaunay triangulation of the cone set, blue<->yellow edges as track gates,
/// then a direction-aware walk over the gate midpoints.
class UnknownPlanner : public Planner
{
public:
  explicit UnknownPlanner(UnknownPlannerParams params = {});

  std::optional<Plan> plan(
    const std::vector<Observation> & observations, PlanDebug * debug) override;
  void reset() override;

  bool closed() const {return closed_;}

private:
  /// Centre of the start line: the big orange cones around the map origin.
  std::optional<Vec2> startGate(const std::vector<Observation> & observations) const;

  /// Driving direction at the start, from the FS colour convention (blue on the
  /// left, yellow on the right). Falls back to the map x axis, which is the
  /// car's initial heading, when only one colour is in view.
  Vec2 forwardAtStart(
    const std::vector<Observation> & observations, Vec2 start) const;

  std::vector<Vec2> gateMidpoints(
    const std::vector<Observation> & observations, PlanDebug * debug) const;

  std::vector<Vec2> trace(
    const std::vector<Vec2> & midpoints, Vec2 start, Vec2 forward) const;

  bool closureFilter(bool detected);

  UnknownPlannerParams params_;
  bool closed_{false};
  int closure_votes_{0};
};

}  // namespace mirena::planning
