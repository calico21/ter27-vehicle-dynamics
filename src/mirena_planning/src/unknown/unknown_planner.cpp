#include "mirena_planning/unknown/unknown_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <utility>

namespace mirena::planning
{
namespace
{

/// One edge of the triangulation, considered as a way through the track.
struct GateEdge
{
  Vec2 mid{};
  double width{0.0};
  int a{-1}, b{-1};
};

/// Every edge of the triangulation, deduplicated -- adjacent triangles share an
/// edge, so each one is found twice.
///
/// Nothing is thrown away here. Whether an edge is a gate depends on where the
/// car is and which way it is going, so it is not something this function could
/// answer; the search answers it.
std::vector<GateEdge> collectEdges(const std::vector<Vec2> & cones, PlanDebug * debug)
{
  const bool verbose = debug && debug->verbose;
  std::vector<GateEdge> edges;
  std::unordered_set<std::int64_t> seen;

  for (const Triangle & tri : delaunay(cones)) {
    for (int e = 0; e < 3; ++e) {
      int a = tri[e], b = tri[(e + 1) % 3];
      if (a > b) {std::swap(a, b);}
      if (!seen.insert((static_cast<std::int64_t>(a) << 32) | static_cast<std::uint32_t>(b)).second) {
        continue;
      }
      if (verbose) {debug->edges.emplace_back(cones[a], cones[b]);}
      edges.push_back({(cones[a] + cones[b]) * 0.5, distance(cones[a], cones[b]), a, b});
    }
  }
  return edges;
}

}  // namespace

UnknownPlanner::UnknownPlanner(UnknownPlannerParams params)
: params_(params) {}

void UnknownPlanner::reset()
{
  closed_ = false;
  closure_votes_ = 0;
}

std::optional<Plan> UnknownPlanner::plan(
  const std::vector<Observation> & observations, PlanDebug * debug)
{
  std::vector<Vec2> cones;
  cones.reserve(observations.size());
  for (const Observation & o : observations) {cones.push_back(o.p);}
  if (cones.size() < 3) {return std::nullopt;}

  const Vec2 start = startGate(observations);
  if (debug) {
    debug->start = start;
    debug->has_start = true;
    if (debug->verbose) {debug->cones = cones;}
  }

  std::vector<Vec2> path = trace(cones, start, debug);
  if (path.size() < 2) {return std::nullopt;}

  bool detected = false;
  if (static_cast<int>(path.size()) > params_.min_path_points) {
    double travelled = 0.0, furthest = 0.0;
    for (std::size_t i = 0; i < path.size(); ++i) {
      furthest = std::max(furthest, distance(path[i], start));
      if (i > 0) {travelled += distance(path[i - 1], path[i]);}
    }
    // Both halves matter. Ending where it began is not a lap on its own -- an
    // excursion out along one boundary and back along the other does that too --
    // so the path also has to be much longer than the ground it covered.
    detected = distance(path.back(), start) < params_.closure_radius &&
      travelled > params_.min_loop_ratio * furthest;
  }
  closed_ = closureFilter(detected);

  // Drop a trailing sample sitting on the start gate: the controller closes the
  // loop itself, and a duplicate would give its periodic spline a zero-length
  // seam segment.
  if (closed_ && path.size() > 2 && distance(path.back(), start) < 1.0) {
    path.pop_back();
  }

  Plan plan;
  plan.gates = buildGates(path, closed_);
  plan.closed = closed_;
  plan.start = start;
  plan.finish = closed_ ? start : path.back();
  return plan;
}

Vec2 UnknownPlanner::startGate(const std::vector<Observation> & observations) const
{
  Vec2 sum{};
  int count = 0;
  for (const Observation & o : observations) {
    if (o.size != ConeSize::Big || norm(o.p) > params_.start_gate_radius) {continue;}
    sum = sum + o.p;
    ++count;
  }
  // Four big cones mark the line, but two are enough to centre it.
  return count >= 2 ? sum * (1.0 / count) : Vec2{};
}

std::vector<Vec2> UnknownPlanner::trace(
  const std::vector<Vec2> & cones, Vec2 start, PlanDebug * debug) const
{
  const std::vector<GateEdge> edges = collectEdges(cones, debug);
  if (edges.empty()) {return {};}

  std::vector<Vec2> midpoints;
  midpoints.reserve(edges.size());
  for (const GateEdge & e : edges) {midpoints.push_back(e.mid);}
  const Grid grid(midpoints, params_.max_gate_gap);

  // Spent cones, not spent gates. A cone is one side of the track at one place
  // along it; it cannot also be the other side somewhere else. So driving
  // through a gate uses both of its cones up, and every other edge touching
  // either of them stops being a way through.
  //
  // This is what rules out the zigzag, and it rules it out rather than merely
  // pricing it: the edge running down the boundary from the gate just driven
  // through shares a cone with it, so it is not an option the search has to
  // weigh and reject -- it is not there. The diagonals leaning back the way the
  // car came go the same way. What survives is the gates ahead, the diagonal
  // that bridges an unmapped cone among them, because that one is made of two
  // cones the car has not spent.
  std::vector<char> spent(cones.size(), 0);

  /// What driving from here through one gate is worth: one gate, less what it
  /// cost to turn to reach it. Anything not worth driving is worth nothing.
  ///
  /// Two things make an edge not a gate at all, and both are here rather than in
  /// the triangulation, because both depend on where the car is standing. An
  /// edge narrower than the track is two cones on the same side of it. A gate
  /// further off than gates are ever set apart is not the next one.
  const auto worth = [&](Vec2 position, Vec2 heading, int index) {
      const GateEdge & gate = edges[index];
      if (spent[gate.a] || spent[gate.b]) {return 0.0;}
      if (gate.width < params_.min_gate_width - params_.width_tolerance ||
        gate.width > params_.max_gate_width)
      {
        return 0.0;
      }

      const Vec2 travel = gate.mid - position;
      const double reach = norm(travel);
      if (reach < 1e-6 || reach > params_.max_gate_gap) {return 0.0;}

      // 0 for carrying straight on, 1 for a right angle, 2 for doubling back, so
      // a car being asked to reverse up its own track is priced out of it
      // without having to be told separately.
      const double turn = 1.0 - dot(travel * (1.0 / reach), heading);
      return 1.0 - params_.curvature_weight * turn -
             params_.reach_weight * (reach / params_.max_gate_gap);
    };

  /// The best that can be earned in the next `depth` steps from this pose.
  ///
  /// The recursion is the point. What a step is worth is what it earns now plus
  /// the best of what it leaves available, so a step onto the wrong boundary --
  /// cheap to take, expensive to leave, because everything after it has to turn
  /// back or dead-end -- loses to the one that keeps the track open. Judged one
  /// step at a time, the two are indistinguishable.
  const std::function<std::pair<int, double>(Vec2, Vec2, int)> look =
    [&](Vec2 position, Vec2 heading, int depth) -> std::pair<int, double> {
      std::pair<int, double> best{-1, 0.0};
      if (depth <= 0) {return best;}

      grid.forEachNear(
        position, params_.max_gate_gap, [&](int index) {
          const double value = worth(position, heading, index);
          if (value <= 0.0) {return;}

          const GateEdge & gate = edges[index];
          spent[gate.a] = spent[gate.b] = 1;
          const Vec2 onward = normalized(gate.mid - position);
          const double total = value + look(gate.mid, onward, depth - 1).second;
          spent[gate.a] = spent[gate.b] = 0;

          if (total > best.second) {best = {index, total};}
        });
      return best;
    };

  // The map frame is anchored at the car's start pose, so its x axis is the
  // direction the car was pointing on the line. That is the only bearing the
  // search needs; the first gate it accepts refines it.
  Vec2 position = start;
  Vec2 heading{1.0, 0.0};
  std::vector<Vec2> path{start};
  bool away = false;

  while (static_cast<int>(path.size()) < params_.max_path_points) {
    const auto [index, value] = look(position, heading, params_.lookahead);
    if (index < 0) {break;}

    spent[edges[index].a] = spent[edges[index].b] = 1;
    heading = normalized(edges[index].mid - position);
    position = edges[index].mid;
    path.push_back(position);
    if (debug && debug->verbose) {
      debug->gate_edges.emplace_back(cones[edges[index].a], cones[edges[index].b]);
    }

    // One lap, and stop. Coming back to the line having been away from it is the
    // end of the track, and carrying on from there starts a second lap on top of
    // the first -- which is worse than merely redundant, because every gate of
    // the real centreline has already been driven through, so the second lap is
    // laid down on whatever is left over: the edges running along the boundaries.
    if (distance(position, start) > params_.closure_radius) {
      away = true;
    } else if (away) {
      break;
    }
  }

  if (debug && debug->verbose) {debug->midpoints = path;}
  return path;
}

bool UnknownPlanner::closureFilter(bool detected)
{
  if (detected == closed_) {
    closure_votes_ = 0;
  } else if (++closure_votes_ >= params_.closure_hysteresis) {
    closed_ = detected;
    closure_votes_ = 0;
  }
  return closed_;
}

}  // namespace mirena::planning
