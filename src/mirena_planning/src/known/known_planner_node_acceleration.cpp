// Track planner for the missions whose layout is surveyed in advance: skidpad,
// acceleration and brake_test.
//
// Loads the mission's template from resource/<mission>.json and registers it
// against the mapped cones with RANSAC + ICP, so the centreline comes from the
// rulebook geometry rather than from whatever has been seen so far.
//
// Two loops run on their own callback groups: planning, which is the expensive
// one, and the heartbeat, which must stay on time because the supervisor reads
// liveness from it. The map subscription only parks the latest message, so a
// burst of map updates can never queue up planning work.
//
// This node and mirena_planning are alternative occupants of one slot: exactly
// one runs per mission, under the node name the supervisor monitors. They are
// deliberately independent of each other -- same topics, same contract, no
// shared code.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nlohmann/json.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "mirena_common/msg/entity_list.hpp"
#include "mirena_common/msg/heartbeat.hpp"
#include "mirena_common/msg/track.hpp"

#include "mirena_planning/known/known_planner_acceleration.hpp"
#include "mirena_planning/planner.hpp"
#include "mirena_planning/known/track_template.hpp"

namespace mirena::planning
{

using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;

namespace
{

constexpr char kMapTopic[] = "slam/map";
constexpr char kTrackTopic[] = "planning/track";
constexpr char kHealthTopic[] = "health";
constexpr char kRegistrationTopic[] = "planning/debug/registration";

using Rgba = std::array<float, 4>;
constexpr Rgba kGrey{0.45f, 0.45f, 0.45f, 0.35f};
constexpr Rgba kGreen{0.15f, 0.9f, 0.3f, 0.9f};
constexpr Rgba kRed{1.0f, 0.2f, 0.2f, 0.9f};
constexpr Rgba kWhite{1.0f, 1.0f, 1.0f, 0.9f};

geometry_msgs::msg::Point toPoint(Vec2 v)
{
  geometry_msgs::msg::Point p;
  p.x = v.x;
  p.y = v.y;
  return p;
}

std_msgs::msg::ColorRGBA toColor(Rgba c)
{
  std_msgs::msg::ColorRGBA out;
  out.r = c[0]; out.g = c[1]; out.b = c[2]; out.a = c[3];
  return out;
}

/// Nothing here reads cone colour, so the marker colour says the one thing that
/// is actually known: orange for the big cones of the start line, grey for the
/// rest.
Rgba coneColor(ConeSize s)
{
  return s == ConeSize::Big ? Rgba{1.0f, 0.35f, 0.0f, 0.9f} : kGrey;
}

Marker makeMarker(
  const std_msgs::msg::Header & header, const char * ns, std::int32_t type,
  double scale, Rgba color)
{
  Marker m;
  m.header = header;
  m.ns = ns;
  m.id = 0;
  m.type = type;
  m.action = Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.scale.x = m.scale.y = m.scale.z = scale;
  m.color = toColor(color);
  return m;
}

void addSegments(Marker & m, const std::vector<std::pair<Vec2, Vec2>> & segments)
{
  m.points.reserve(m.points.size() + 2 * segments.size());
  for (const auto & [a, b] : segments) {
    m.points.push_back(toPoint(a));
    m.points.push_back(toPoint(b));
  }
}

void addPoints(Marker & m, const std::vector<Vec2> & points)
{
  m.points.reserve(m.points.size() + points.size());
  for (const Vec2 & p : points) {m.points.push_back(toPoint(p));}
}

}  // namespace

class KnownPlannerNode : public rclcpp::Node
{
public:
  KnownPlannerNode()
  : rclcpp::Node("mirena_planning")
  {
    declareParameters();

    mission_ = get_parameter("mission").as_string();
    RCLCPP_INFO(get_logger(), "Planning mission '%s'", mission_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "params: plan_period=%.3f heartbeat_period=%.3f republish_tolerance=%.2f publish_debug=%d",
      get_parameter("plan_period").as_double(),
      get_parameter("heartbeat_period").as_double(),
      get_parameter("republish_tolerance").as_double(),
      get_parameter("publish_debug").as_bool() ? 1 : 0);

    // The mission launch file picks the planner, so the two should never
    // disagree -- but a hand-rolled `ros2 run` can still get it wrong.
    //
    // brake_test drives the acceleration straight, so it is surveyed like the
    // other two and registers resource/brake_test.json.
    if (mission_ != "skidpad" && mission_ != "acceleration" && mission_ != "brake_test") {
      RCLCPP_ERROR(
        get_logger(),
        "Mission '%s' has no surveyed layout. Launch mirena_planning for it.",
        mission_.c_str());
      throw std::runtime_error("wrong planner node for this mission");
    }

    republish_tolerance_ = get_parameter("republish_tolerance").as_double();
    fallback_frame_ = get_parameter("fallback_frame").as_string();
    publish_debug_ = get_parameter("publish_debug").as_bool();

    planner_ = makePlanner();
    if (!planner_) {throw std::runtime_error("planner could not be built");}

    plan_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    report_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions map_options;
    map_options.callback_group = plan_group_;
    map_sub_ = create_subscription<mirena_common::msg::EntityList>(
      kMapTopic, 10,
      std::bind(&KnownPlannerNode::mapCallback, this, std::placeholders::_1), map_options);

    // Latched: the track changes rarely once it settles, and a controller that
    // subscribes late (or restarts) should not have to wait for the next
    // material change to learn where the track is.
    track_pub_ = create_publisher<mirena_common::msg::Track>(
      kTrackTopic, rclcpp::QoS(1).transient_local());
    report_pub_ = create_publisher<mirena_common::msg::Heartbeat>(kHealthTopic, 10);
    if (publish_debug_) {
      registration_pub_ = create_publisher<MarkerArray>(kRegistrationTopic, 1);
    }

    plan_timer_ = rclcpp::create_timer(
      this, get_clock(),
      rclcpp::Duration::from_seconds(get_parameter("plan_period").as_double()),
      std::bind(&KnownPlannerNode::planIteration, this), plan_group_);
    report_timer_ = rclcpp::create_timer(
      this, get_clock(),
      rclcpp::Duration::from_seconds(get_parameter("heartbeat_period").as_double()),
      std::bind(&KnownPlannerNode::publishHeartbeat, this), report_group_);
  }

private:
  /// Everything the heartbeat reports, written by the planning loop.
  struct Report
  {
    std::optional<Vec2> start;
    std::size_t gates{0};
    bool closed{false};
    bool planned{false};
    double score{0.0};
    double plan_ms{0.0};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool has_plan{false};
  };

  /// The last track actually put on the wire, so we can tell a real change from
  /// the millimetre jitter of a re-registered centreline.
  struct Published
  {
    std::vector<Gate> gates;
    bool closed{false};
    bool valid{false};
  };

  void declareParameters()
  {
    // Declared and read in the same breath: the mission's values arrive as
    // launch arguments, so they are already in place by construction. Each
    // value below is echoed into the heartbeat indirectly via the planner's
    // score/inliers, so a mismatch between launch file and node shows up as a
    // low score rather than a silent failure.
    const KnownPlannerAccelerationParams k;

    declare_parameter<std::string>("mission", "acceleration");
    declare_parameter<std::string>("fallback_frame", "map");
    declare_parameter<double>("plan_period", 0.1);
    declare_parameter<double>("heartbeat_period", 0.1);
    declare_parameter<bool>("publish_debug", true);
    // Below this the centreline has not meaningfully moved, so republishing it
    // would only make the controller re-fit its spline and shift the arc length
    // its lap counter rides on.
    declare_parameter<double>("republish_tolerance", 0.25);

    // Registration quality thresholds. inlier_distance is the max radius (m)
    // within which a SLAM cone counts as matched to a template cone;
    // min_score/min_inliers gate whether a fit is plausible at all.
    declare_parameter<double>("inlier_distance", k.inlier_distance);
    declare_parameter<double>("min_score", k.min_score);
    declare_parameter<int>("min_inliers", k.min_inliers);
    declare_parameter<int>("icp_iterations", k.icp_iterations);
    declare_parameter<int>("ransac_iterations", k.ransac_iterations);

    // Gate-sequence recovery (used when seeded ICP fails). gate_min/max_width
    // bracket the rule gate width (3 m rule, 5 m cap); the angle tolerance
    // decides if a candidate pair looks like a transverse gate; the sequence
    // tolerances check that three observed gates line up like three template
    // gates; spacing tolerance compares observed vs template gate spacing.
    declare_parameter<double>("gate_min_width", k.gate_min_width);
    declare_parameter<double>("gate_max_width", k.gate_max_width);
    declare_parameter<double>(
      "gate_longitudinal_tolerance", k.gate_longitudinal_tolerance);
    declare_parameter<double>("gate_angle_tolerance", k.gate_angle_tolerance);
    declare_parameter<double>("sequence_line_tolerance", k.sequence_line_tolerance);
    declare_parameter<double>("gate_centerline_tolerance", k.gate_centerline_tolerance);
    declare_parameter<double>("gate_spacing_tolerance", k.gate_spacing_tolerance);

    // Plausibility drift limits vs the nominal template pose: if the fitted
    // start point or heading strays beyond these, the registration is rejected
    // even if the score looks fine.
    declare_parameter<double>("max_position_drift", k.max_position_drift);
    declare_parameter<double>("max_heading_drift", k.max_heading_drift);
  }

  std::unique_ptr<Planner> makePlanner()
  {
    KnownPlannerAccelerationParams p;
    p.inlier_distance = get_parameter("inlier_distance").as_double();
    p.min_score = get_parameter("min_score").as_double();
    p.min_inliers = static_cast<int>(get_parameter("min_inliers").as_int());
    p.icp_iterations = static_cast<int>(get_parameter("icp_iterations").as_int());
    p.ransac_iterations = static_cast<int>(get_parameter("ransac_iterations").as_int());

    // Acceleration gate-sequence recovery parameters.
    p.gate_min_width = get_parameter("gate_min_width").as_double();
    p.gate_max_width = get_parameter("gate_max_width").as_double();
    p.gate_longitudinal_tolerance =
      get_parameter("gate_longitudinal_tolerance").as_double();
    p.gate_angle_tolerance = get_parameter("gate_angle_tolerance").as_double();
    p.sequence_line_tolerance = get_parameter("sequence_line_tolerance").as_double();
    p.gate_centerline_tolerance = get_parameter("gate_centerline_tolerance").as_double();
    p.gate_spacing_tolerance = get_parameter("gate_spacing_tolerance").as_double();

    p.max_position_drift = get_parameter("max_position_drift").as_double();
    p.max_heading_drift = get_parameter("max_heading_drift").as_double();

    // One template per mission, installed with the package. The JSON names the
    // mission and lists the surveyed cones plus the nominal centreline; the
    // planner registers live cones against it, so a missing or malformed file
    // means this node cannot plan at all.
    const std::string path = ament_index_cpp::get_package_share_directory("mirena_planning") +
      "/resource/" + mission_ + ".json";
    try {
      auto track = loadTrackTemplate(path);
      RCLCPP_INFO(
        get_logger(), "Loaded template '%s': %zu cones, %zu path points.",
        track.name.c_str(), track.cones.size(), track.path.size());
      // Watch this line: cones/path counts of 0 mean the JSON loaded but is
      // empty, and the planner will return no plan until it is fixed.
      return std::make_unique<KnownPlannerAcceleration>(std::move(track), p);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "%s", e.what());
      return nullptr;
    }
  }

  void mapCallback(mirena_common::msg::EntityList::SharedPtr msg)
  {
    // Park it and return: planning runs on its own timer, so a fast map
    // publisher cannot build a backlog of expensive callbacks.
    const std::size_t entity_count = msg->entities.size();
    std::lock_guard<std::mutex> lock(map_mutex_);
    pending_map_ = std::move(msg);
    if (!map_seen_) {
      map_seen_ = true;
      RCLCPP_INFO(
        get_logger(), "mapCallback: first map received with %zu entities on topic '%s'",
        entity_count, kMapTopic);
    }
  }

  void planIteration()
  {
    // Guard: the constructor may not have finished building the planner yet
    // (a rejected mission leaves planner_ null), or the track publisher may
    // not exist. Nothing to do until both are ready.
    if (!planner_ || !track_pub_) {return;}

    // Consume the parked map. exchange() leaves pending_map_ null so a burst
    // of map updates can never queue up more than one planning cycle.
    mirena_common::msg::EntityList::SharedPtr map;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      map = std::exchange(pending_map_, nullptr);
    }
    // No new map since the last cycle: normal early-run state, not an error.
    if (!map) {return;}

    // Turn every SLAM cone into an observation. The pipeline never reports a
    // cone colour, so discarding colourless cones would discard all of them.
    std::vector<Observation> observations;
    observations.reserve(map->entities.size());
    for (const auto & entity : map->entities) {
      const ConeSize size = coneSizeFromString(entity.type);
      observations.push_back(
        Observation{{entity.position.x, entity.position.y}, size});
    }

    std_msgs::msg::Header header = map->header;
    if (header.frame_id.empty()) {header.frame_id = fallback_frame_;}

    RCLCPP_INFO(
      get_logger(), "planIteration: %zu observations frame='%s'",
      observations.size(), header.frame_id.c_str());

    // Only collect the full debug geometry (correspondences, fitted template,
    // rejected points) when someone is actually listening on the debug topic.
    PlanDebug debug;
    debug.verbose = debugWanted();

    // Time the plan so the heartbeat can report plan_ms; this is also the
    // number to watch if the planner starts missing its plan_period.
    const auto started = std::chrono::steady_clock::now();
    const std::optional<Plan> plan = planner_->plan(observations, &debug);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();

    RCLCPP_INFO(
      get_logger(),
      "planIteration: planner result planned=%d score=%.2f inliers-via-debug n/a plan_ms=%.1f",
      plan.has_value() ? 1 : 0, debug.score, elapsed_ms);

    if (debug.verbose) {publishDebug(header, debug);}

    // Record timing/score under lock so the heartbeat can read a consistent
    // snapshot without blocking the planning loop for long.
    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      report_.plan_ms = elapsed_ms;
      report_.score = debug.score;
      report_.planned = plan.has_value();
      if (plan) {
        report_.start = plan->start;
        report_.gates = plan->gates.size();
        report_.closed = plan->closed;
        report_.stamp = now();
        report_.has_plan = true;
      }
    }

    // Bail out on the normal not-ready states: no registration yet, fewer
    // than two gates (not a track), or a centreline that has not moved by more
    // than republish_tolerance (millimetre jitter from re-registration must
    // not make the controller re-fit its spline).
    if (!plan) {
      RCLCPP_WARN(
        get_logger(), "planIteration: planner returned no plan (score=%.2f)", debug.score);
      return;
    }
    if (plan->gates.size() < 2) {
      RCLCPP_WARN(
        get_logger(),
        "planIteration: plan has %zu gates (< 2), not a track yet, not publishing",
        plan->gates.size());
      return;
    }
    const bool changed = trackChanged(*plan);
    if (!changed) {
      RCLCPP_INFO(
        get_logger(),
        "planIteration: centreline within republish tolerance %.2f m, skipping publish",
        republish_tolerance_);
      return;
    }

    // Build and publish the track. Latched QoS means a late-joining controller
    // gets the last good track immediately.
    RCLCPP_INFO(
      get_logger(),
      "planIteration: publishing track %zu gates closed=%d score=%.2f plan_ms=%.1f",
      plan->gates.size(), plan->closed ? 1 : 0, debug.score, elapsed_ms);

    mirena_common::msg::Track track;
    track.header = header;
    track.is_closed = plan->closed;
    track.gates.reserve(plan->gates.size());
    for (const Gate & gate : plan->gates) {
      mirena_common::msg::Gate out;
      out.x = gate.p.x;
      out.y = gate.p.y;
      out.psi = gate.psi;
      track.gates.push_back(out);
    }
    track_pub_->publish(std::move(track));
    RCLCPP_INFO(get_logger(), "planIteration: track published on topic '%s'", kTrackTopic);

    // Remember what went out so trackChanged() can tell a real move from jitter.
    published_.gates = plan->gates;
    published_.closed = plan->closed;
    published_.valid = true;
  }

  /// True when the new centreline differs from the published one by more than
  /// the planner's own noise. A change here triggers a republish on
  /// planning/track; otherwise the old track stands so the controller's spline
  /// fit (and the lap counter riding on its arc length) is not disturbed.
  bool trackChanged(const Plan & plan) const
  {
    // Never published anything yet, or the track topology changed (open vs
    // closed): always republish.
    if (!published_.valid || published_.closed != plan.closed) {return true;}
    // Different gate count means a different registration: republish.
    if (published_.gates.size() != plan.gates.size()) {return true;}
    // Same number of gates: any gate that moved by more than the tolerance
    // counts as a real change.
    for (std::size_t i = 0; i < plan.gates.size(); ++i) {
      if (distance(plan.gates[i].p, published_.gates[i].p) > republish_tolerance_) {return true;}
    }
    // Everything within noise: do not republish.
    return false;
  }

  bool debugWanted() const
  {
    return registration_pub_ && registration_pub_->get_subscription_count() > 0;
  }

  void publishDebug(const std_msgs::msg::Header & header, const PlanDebug & debug)
  {
    // Visualisation on planning/debug/registration, only when a subscriber is
    // actually listening (see debugWanted()). Each marker answers one question:
    //
    //   template        -- where the planner thinks the surveyed cones are, in map frame
    //   template_path   -- the surveyed centreline path, transformed the same way
    //   correspondences -- observation -> matched template cone (green lines)
    //   rejected        -- observations with no template match (red points)
    //   status          -- one-line text: REGISTERED vs SEARCHING, score, ICP
    //                    iterations, whether the seed came from RANSAC, and the
    //                    fitted translation/heading

    Marker clear;
    clear.header = header;
    clear.action = Marker::DELETEALL;

    MarkerArray array;
    array.markers.push_back(std::move(clear));

    Marker cones = makeMarker(header, "template", Marker::SPHERE_LIST, 0.35, kWhite);
    for (const Observation & cone : debug.fitted_template) {
      cones.points.push_back(toPoint(cone.p));
      cones.colors.push_back(toColor(coneColor(cone.size)));
    }
    array.markers.push_back(std::move(cones));

    Marker path = makeMarker(header, "template_path", Marker::LINE_STRIP, 0.10, kWhite);
    addPoints(path, debug.fitted_path);
    array.markers.push_back(std::move(path));

    Marker inliers = makeMarker(header, "correspondences", Marker::LINE_LIST, 0.03, kGreen);
    addSegments(inliers, debug.correspondences);
    array.markers.push_back(std::move(inliers));

    Marker rejected = makeMarker(header, "rejected", Marker::SPHERE_LIST, 0.3, kRed);
    addPoints(rejected, debug.rejected);
    array.markers.push_back(std::move(rejected));

    Marker status = makeMarker(
      header, "status", Marker::TEXT_VIEW_FACING, 0.6,
      debug.registered ? kGreen : kRed);
    status.pose.position.z = 2.0;
    char text[160];
    std::snprintf(
      text, sizeof(text), "%s  score %.2f  icp %d  %s\nfit  x %.2f  y %.2f  psi %.3f",
      debug.registered ? "REGISTERED" : "SEARCHING", debug.score, debug.iterations,
      debug.from_ransac ? "(ransac)" : "(seeded)",
      debug.fit.t.x, debug.fit.t.y, debug.fit.psi);
    status.text = text;
    array.markers.push_back(std::move(status));

    registration_pub_->publish(std::move(array));
    RCLCPP_INFO(
      get_logger(),
      "publishDebug: sent registration marker array (registered=%d score=%.2f iterations=%d from_ransac=%d)",
      debug.registered ? 1 : 0, debug.score, debug.iterations, debug.from_ransac ? 1 : 0);
  }

  void publishHeartbeat()
  {
    if (!report_pub_) {return;}

    Report report;
    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      report = report_;
    }

    nlohmann::json json;
    // Kept as plain numbers (and null before the first plan) so any consumer can
    // read the gate straight out of the heartbeat JSON.
    json["start_point"] = nullptr;
    if (report.start) {
      json["start_point"] = {{"x", report.start->x}, {"y", report.start->y}, {"z", 0.0}};
    }
    json["mission"] = mission_;
    json["gates"] = report.gates;
    json["closed"] = report.closed;
    json["plan_ms"] = report.plan_ms;
    json["planned"] = report.planned;
    json["score"] = report.score;
    json["plan_age_s"] = report.has_plan ?
      (now() - report.stamp).seconds() : std::numeric_limits<double>::quiet_NaN();

    // The heartbeat JSON is the diagnosis channel on a car with no screen: the
    // supervisor reads planned/score/plan_ms/plan_age_s from it. Numbers are
    // plain so any consumer can read the gate straight out of the JSON.
    mirena_common::msg::Heartbeat msg;
    msg.node_name = get_name();
    msg.timestamp = now();
    msg.report = json.dump();
    const double plan_age_s = report.has_plan ? (now() - report.stamp).seconds() : -1.0;
    RCLCPP_INFO(
      get_logger(),
      "heartbeat: planned=%d gates=%zu score=%.2f plan_ms=%.1f plan_age_s=%.2f mission=%s",
      report.planned ? 1 : 0, report.gates, report.score, report.plan_ms, plan_age_s,
      mission_.c_str());
    report_pub_->publish(msg);
  }

  // Configuration
  std::string mission_;
  std::string fallback_frame_{"map"};
  double republish_tolerance_{0.25};
  bool publish_debug_{true};

  std::unique_ptr<Planner> planner_;

  rclcpp::CallbackGroup::SharedPtr plan_group_;
  rclcpp::CallbackGroup::SharedPtr report_group_;

  rclcpp::Subscription<mirena_common::msg::EntityList>::SharedPtr map_sub_;
  rclcpp::Publisher<mirena_common::msg::Track>::SharedPtr track_pub_;
  rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr report_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr registration_pub_;

  rclcpp::TimerBase::SharedPtr plan_timer_;
  rclcpp::TimerBase::SharedPtr report_timer_;

  std::mutex map_mutex_;
  mirena_common::msg::EntityList::SharedPtr pending_map_;
  bool map_seen_{false};

  // Only the planning loop touches this one.
  Published published_;

  std::mutex report_mutex_;
  Report report_;
};

}  // namespace mirena::planning

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<mirena::planning::KnownPlannerNode> node;
  try {
    node = std::make_shared<mirena::planning::KnownPlannerNode>();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("mirena_planning"), "Startup failed: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
