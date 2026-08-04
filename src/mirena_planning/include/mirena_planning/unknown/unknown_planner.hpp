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
  /// Narrowest edge that can be a way through the track, in metres. The track
  /// is 3 m across, so anything narrower is two cones on the same side of it.
  double min_gate_width{3.0};
  /// How far under that a measured width may fall and still count.
  ///
  /// Not slack for its own sake. What gets measured is not the track, it is two
  /// mapped cones, and a gate pegged out at 3.000 m arrives here measuring
  /// 2.9992. Tested against 3 exactly, a third of the real gates on the sim's own
  /// exported tracks are thrown away and the trace stops at the first of them.
  double width_tolerance{0.1};
  /// Widest edge that can be a way through the track, in metres.
  ///
  /// Set to let the diagonals through, not to keep them out. Between two gates
  /// the triangulation always draws the two crossing diagonals, and they are
  /// real gates: their midpoints sit on the centreline, and where a cone is
  /// missing they are the only gate there is. On the sim's five exported tracks
  /// with 6% of cones unmapped, admitting them traced 77% of the lap and
  /// excluding them traced 12%.
  ///
  /// What the ceiling is for is everything longer than a diagonal -- the chord
  /// across the inside of a hairpin, the edge reaching clean across a corner.
  /// Those are not ways through anything. Uncapped, the same five tracks put the
  /// centreline 24 m off the track.
  double max_gate_width{5.0};

  /// Furthest apart two consecutive gates may be, in metres.
  ///
  /// Twice the 5 m cones are set out at, and the doubling is the point. Five
  /// metres is how far apart *cones* are; gates are only that far apart while
  /// every cone is in the map. Lose one and the next gate the car has not
  /// already spent is two stations away, so a 5 m reach cannot get there and the
  /// trace stops at the hole. On the five exported tracks with 6% of cones
  /// unmapped, 5 m traced 6% of the lap and 10 m traced 93%.
  ///
  /// Reaching this far is only safe because reach_weight makes it unattractive.
  /// Without that the search strides whenever it can, and a 10 m gap put the
  /// centreline 1.5 m off on a complete map; with it the trace steps through the
  /// same 70 gates a 5 m cap does, and spends the extra reach only at holes.
  double max_gate_gap{10.0};

  /// What bending the centreline costs, against the one gate a step is worth.
  ///
  /// A step earns 1 for the gate it drives through and pays this many times the
  /// turn it took to get there, so at 1.0 a step that turns a right angle earns
  /// nothing and anything sharper earns less than not moving. That is all there
  /// is to whether a step is worth taking -- no separate threshold, because a
  /// step nobody would drive is already worth less than staying put.
  double curvature_weight{1.0};
  /// What reaching costs, per full gap of reach, against that same one gate.
  ///
  /// The gap has to be generous, because a cone perception missed puts the next
  /// gate two stations away instead of one. But a generous gap with nothing to
  /// discourage it means the search reaches for no reason: a gate three along is
  /// worth exactly what the next one is worth, so it strides past gates it
  /// should have driven through and lands off the line. This makes the nearer
  /// gate the better one when both are otherwise equal, so the reach is there
  /// when it is needed and unused when it is not.
  double reach_weight{0.5};

  /// How many gates ahead the search looks before committing to the next one.
  ///
  /// This is what stops the tracer being fooled by a gate that looks good on its
  /// own. A step onto the wrong boundary is always cheap to take and always
  /// expensive to leave -- everything after it has to turn back, or dead-ends --
  /// and the only way to see that is to look past it. One is a greedy walk,
  /// which commits to its first mistake. The cost of looking is exponential in
  /// this, so it wants to stay small.
  ///
  /// Odd on purpose. A zigzag between the centreline and a boundary repeats
  /// every two steps, so an even horizon ends where it started and the detour
  /// looks free; an odd one always ends mid-zigzag and has to pay for it. On the
  /// sim's five exported tracks a horizon of four put the centreline 1.7 m off
  /// the track, where three and five both hold it inside 5 cm.
  int lookahead{5};

  /// Distance to the start gate that counts as "back at the line". A tolerance
  /// on SLAM's idea of where the line is, not a property of the track.
  double closure_radius{8.0};
  /// How much longer a lap is than the ground it covers: the distance driven
  /// over the furthest the path ever got from the line.
  ///
  /// This is what stops a there-and-back being called a lap. Run out along one
  /// boundary and back along the other and the path is twice its own reach;
  /// anything that genuinely goes round is more, a circle being pi times its
  /// own. Scale free, so it says the same thing about a short circuit and a long
  /// one.
  double min_loop_ratio{2.5};
  /// A path shorter than this cannot be a closed lap.
  int min_path_points{15};
  /// Consecutive agreeing detections needed to flip the published closed flag.
  /// Without it the flag chatters and the controller keeps re-fitting its spline
  /// between periodic and open.
  int closure_hysteresis{5};
  /// Hard cap on the walk, so a pathological cone set cannot spin forever.
  int max_path_points{400};

  /// Radius around the map origin searched for the big cones of the start line.
  /// The map frame is anchored at the car's start pose, so the line is at the
  /// origin; this is what stops an autocross finish gate being averaged in.
  double start_gate_radius{10.0};
  /// Cones below this confidence are ignored.
  double min_confidence{0.0};
};

/// Colourless centreline finder: Delaunay-triangulate the cones, then drive
/// through as many of the resulting edges as possible, as straight as possible.
///
/// That sentence is the whole planner. A step through a gate is worth one gate
/// less what it cost to turn, and the search is a recursion -- what a step is
/// worth is what it earns now plus the best that can be earned after it. So a
/// step that looks fine on its own and dead-ends a few gates later loses to one
/// that looks slightly worse and opens the track up, which is exactly the
/// judgement a greedy tracer cannot make.
///
/// There is no test for what counts as a gate. Every edge of the triangulation
/// is a candidate, including the ones running down a boundary, and they lose on
/// their own merits: an edge along the track leads to a midpoint off to one
/// side, and everything after it has to turn back. A wrong step is not rejected,
/// it is simply outbid.
class UnknownPlanner : public Planner
{
public:
  explicit UnknownPlanner(UnknownPlannerParams params = {});

  std::optional<Plan> plan(
    const std::vector<Observation> & observations, PlanDebug * debug) override;
  void reset() override;

  bool closed() const {return closed_;}

private:
  /// Where the track starts. With no big cones mapped the origin is the honest
  /// answer rather than a guess: the map frame is anchored at the car's start
  /// pose, and the car starts on the line.
  Vec2 startGate(const std::vector<Observation> & observations) const;

  std::vector<Vec2> trace(
    const std::vector<Vec2> & cones, Vec2 start, PlanDebug * debug) const;

  bool closureFilter(bool detected);

  UnknownPlannerParams params_;
  bool closed_{false};
  int closure_votes_{0};
};

}  // namespace mirena::planning
