#include "mirena_planning/unknown_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

namespace mirena::planning
{

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
  const auto start = startGate(observations);
  if (!start) {return std::nullopt;}
  if (debug) {
    debug->start = *start;
    debug->has_start = true;
  }

  const std::vector<Vec2> midpoints = gateMidpoints(observations, debug);
  if (midpoints.empty()) {return std::nullopt;}

  std::vector<Vec2> path = trace(midpoints, *start, forwardAtStart(observations, *start));
  if (path.size() < 2) {return std::nullopt;}

  bool detected = false;
  if (static_cast<int>(path.size()) > params_.min_path_points) {
    double furthest = 0.0;
    for (const Vec2 & p : path) {furthest = std::max(furthest, distance(p, *start));}
    // Both halves matter: a stub that never leaves the line would otherwise
    // report a closed lap purely because it ends where it began.
    detected = distance(path.back(), *start) < params_.closure_radius &&
      furthest > 2.0 * params_.closure_radius;
  }
  closed_ = closureFilter(detected);

  // Drop a trailing sample sitting on the start gate: the controller closes the
  // loop itself, and a duplicate would give its periodic spline a zero-length
  // seam segment.
  if (closed_ && path.size() > 2 && distance(path.back(), *start) < 1.0) {
    path.pop_back();
  }

  Plan plan;
  plan.gates = buildGates(path, closed_);
  plan.closed = closed_;
  plan.start = *start;
  plan.finish = closed_ ? *start : path.back();
  return plan;
}

std::optional<Vec2> UnknownPlanner::startGate(const std::vector<Observation> & observations) const
{
  // Prefer the big orange cones that mark the line; fall back to small orange
  // only if none were mapped.
  for (const Cone wanted : {Cone::BigOrange, Cone::Orange}) {
    Vec2 sum{};
    int count = 0;
    for (const Observation & o : observations) {
      if (o.color != wanted || o.confidence < params_.min_confidence) {continue;}
      if (norm(o.p) > params_.start_gate_radius) {continue;}
      sum = sum + o.p;
      ++count;
    }
    if (count >= 2) {return sum * (1.0 / count);}
  }
  return std::nullopt;
}

Vec2 UnknownPlanner::forwardAtStart(
  const std::vector<Observation> & observations, Vec2 start) const
{
  const Observation * blue = nullptr;
  const Observation * yellow = nullptr;
  double blue_d = std::numeric_limits<double>::infinity();
  double yellow_d = blue_d;

  for (const Observation & o : observations) {
    if (o.confidence < params_.min_confidence) {continue;}
    const double d = distance(o.p, start);
    if (o.color == Cone::Blue && d < blue_d) {blue_d = d; blue = &o;}
    if (o.color == Cone::Yellow && d < yellow_d) {yellow_d = d; yellow = &o;}
  }

  if (blue && yellow) {
    // blue -> yellow points to the right of travel, so its +90 degree rotation
    // is forward.
    const Vec2 forward = normalized(perp(yellow->p - blue->p));
    if (squaredNorm(forward) > 0.5) {return forward;}
  }
  return {1.0, 0.0};
}

std::vector<Vec2> UnknownPlanner::gateMidpoints(
  const std::vector<Observation> & observations, PlanDebug * debug) const
{
  std::vector<Vec2> cones;
  std::vector<Cone> colors;
  for (const Observation & o : observations) {
    if (o.confidence < params_.min_confidence) {continue;}
    if (o.color != Cone::Blue && o.color != Cone::Yellow) {continue;}
    cones.push_back(o.p);
    colors.push_back(o.color);
  }
  const bool verbose = debug && debug->verbose;
  if (verbose) {debug->cones = cones;}
  if (cones.size() < 3) {return {};}

  std::vector<Vec2> raw;
  for (const Triangle & tri : delaunay(cones)) {
    for (int e = 0; e < 3; ++e) {
      const int a = tri[e], b = tri[(e + 1) % 3];
      if (verbose && a < b) {debug->edges.emplace_back(cones[a], cones[b]);}
      if (colors[a] == colors[b]) {continue;}
      if (distance(cones[a], cones[b]) > params_.max_gate_width) {continue;}
      if (verbose && a < b) {debug->gate_edges.emplace_back(cones[a], cones[b]);}
      raw.push_back((cones[a] + cones[b]) * 0.5);
    }
  }

  // Adjacent triangles share an edge, so every gate is found twice; merge on a
  // grid of merge_radius rather than trusting exact equality.
  std::vector<Vec2> midpoints;
  std::unordered_set<std::int64_t> seen;
  const double inv = 1.0 / std::max(params_.merge_radius, 1e-6);
  for (const Vec2 & m : raw) {
    const auto kx = static_cast<std::int64_t>(std::llround(m.x * inv));
    const auto ky = static_cast<std::int64_t>(std::llround(m.y * inv));
    if (seen.insert((kx << 32) ^ (ky & 0xffffffff)).second) {midpoints.push_back(m);}
  }
  if (verbose) {debug->midpoints = midpoints;}
  return midpoints;
}

std::vector<Vec2> UnknownPlanner::trace(
  const std::vector<Vec2> & midpoints, Vec2 start, Vec2 forward) const
{
  const Grid grid(midpoints, params_.max_gate_gap);
  std::vector<bool> used(midpoints.size(), false);

  std::vector<Vec2> path{start};
  Vec2 current = start;
  Vec2 heading = forward;

  // Each step takes the unvisited midpoint that is both close and roughly
  // straight ahead. The heading term is what keeps the walk from cutting across
  // the track or doubling back, which plain nearest-neighbour does freely.
  while (true) {
    int best = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    grid.forEachNear(
      current, params_.max_gate_gap, [&](int i) {
        if (used[i]) {return;}
        const Vec2 step = midpoints[i] - current;
        const double d = norm(step);
        if (d > params_.max_gate_gap || d < 1e-6) {return;}

        double cost = d;
        if (squaredNorm(heading) > 0.5) {
          const double alignment = dot(step * (1.0 / d), heading);
          if (alignment < params_.min_progress_cos) {return;}
          cost = d * (1.0 + params_.turn_penalty * (1.0 - alignment));
        }
        if (cost < best_cost) {best_cost = cost; best = i;}
      });

    if (best < 0) {break;}
    heading = normalized(midpoints[best] - current);
    current = midpoints[best];
    used[best] = true;
    path.push_back(current);
  }

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
