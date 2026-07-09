// SocketCAN bridge node for the TER DBC (hardware).
//
// Implements three paths:
//   * CarControl (ROS) -> DV_Dynamic_Req_1 (CAN)        [output]
//   * MissionInfo (ROS) -> DV_info (CAN)                [output]
//   * DV_system_status (CAN) -> AsInfo (ROS)            [input]
//
// The CAN interface is expected to be already up (bitrate configured, link up);
// bringing it up is the responsibility of the system (e.g. systemd-networkd / ip link).
// CAN (de)serialization is delegated to the cantools-generated C library (ter.h/ter.c).

#include <cstdint>
#include <cstring>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>

#include <unistd.h>
#include <fcntl.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

#include "rclcpp/rclcpp.hpp"

#include "mirena_common/msg/car_control.hpp"
#include "mirena_common/msg/as_info.hpp"
#include "mirena_common/msg/mission_info.hpp"

extern "C" {
#include "ter.h"
}

using namespace std::chrono_literals;
using std::placeholders::_1;

class MirenaCan : public rclcpp::Node
{
public:
  MirenaCan()
  : rclcpp::Node("mirena_can")
  {
    iface_ = declare_parameter<std::string>("can_interface", "can1");

    if (!open_can(iface_)) {
      // Keep the node alive so the rest of the stack still launches; CAN I/O is gated on the fd.
      RCLCPP_ERROR(get_logger(), "CAN bus '%s' unavailable; node running but inactive", iface_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "CAN bus opened on interface '%s'", iface_.c_str());
    }

    control_sub_ = create_subscription<mirena_common::msg::CarControl>(
      "/control", 10, std::bind(&MirenaCan::control_callback, this, _1));

    mission_info_sub_ = create_subscription<mirena_common::msg::MissionInfo>(
      "/system/mission_status", 10, std::bind(&MirenaCan::mission_info_callback, this, _1));

    as_info_pub_ = create_publisher<mirena_common::msg::AsInfo>("/as_status", 10);

    // Poll the socket regularly; reads are non-blocking and drain all pending frames.
    rx_timer_ = create_wall_timer(5ms, std::bind(&MirenaCan::rx_poll, this));
  }

  ~MirenaCan() override { close_can(); }

private:
  // --- CAN socket ---
  bool open_can(const std::string & iface)
  {
    can_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_fd_ < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to create CAN socket: %s", std::strerror(errno));
      return false;
    }

    struct ifreq ifr {};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(can_fd_, SIOCGIFINDEX, &ifr) < 0) {
      RCLCPP_ERROR(get_logger(), "Interface '%s' not found: %s", iface.c_str(), std::strerror(errno));
      close_can();
      return false;
    }

    // Only deliver DV_system_status frames to this socket.
    struct can_filter rfilter[1];
    rfilter[0].can_id = TER_DV_SYSTEM_STATUS_FRAME_ID;
    rfilter[0].can_mask = CAN_SFF_MASK;  // DV_system_status is a standard (11-bit) frame
    if (setsockopt(can_fd_, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter)) < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to set CAN filter: %s", std::strerror(errno));
      close_can();
      return false;
    }

    struct sockaddr_can addr {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(can_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to bind CAN socket: %s", std::strerror(errno));
      close_can();
      return false;
    }

    const int flags = fcntl(can_fd_, F_GETFL, 0);
    fcntl(can_fd_, F_SETFL, flags | O_NONBLOCK);
    return true;
  }

  void close_can()
  {
    if (can_fd_ >= 0) {
      ::close(can_fd_);
      can_fd_ = -1;
    }
  }

  // --- ROS CarControl -> CAN DV_Dynamic_Req_1 ---
  void control_callback(const mirena_common::msg::CarControl::SharedPtr msg)
  {
    if (can_fd_ < 0) {
      return;
    }

    struct ter_dv_dynamic_req_1_t src;
    ter_dv_dynamic_req_1_init(&src);
    src.trq_req = ter_dv_dynamic_req_1_trq_req_encode(static_cast<float>(msg->gas));
    src.steer_angle_req = ter_dv_dynamic_req_1_steer_angle_req_encode(static_cast<float>(msg->steer_angle));

    struct can_frame frame {};
    frame.can_id = TER_DV_DYNAMIC_REQ_1_FRAME_ID;  // standard frame, no flags
    frame.can_dlc = TER_DV_DYNAMIC_REQ_1_LENGTH;

    const int packed = ter_dv_dynamic_req_1_pack(frame.data, &src, sizeof(frame.data));
    if (packed < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to pack DV_Dynamic_Req_1");
      return;
    }

    if (write(can_fd_, &frame, sizeof(frame)) != static_cast<ssize_t>(sizeof(frame))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Failed to write DV_Dynamic_Req_1 to CAN: %s", std::strerror(errno));
    }
  }

  // --- ROS MissionInfo -> CAN DV_info ---
  void mission_info_callback(const mirena_common::msg::MissionInfo::SharedPtr msg)
  {
    if (can_fd_ < 0) {
      return;
    }

    struct ter_dv_info_t src;
    ter_dv_info_init(&src);

    src.mission_status = map_mission_status(msg->status);
    // DV_info packs these signals into limited bit-widths (see ter.c pack): the
    // encode helpers only cast, so clamp to the wire range first.
    src.cones_count_actual =
      ter_dv_info_cones_count_actual_encode(static_cast<float>(std::min<uint32_t>(msg->cones_count_actual, 255u)));
    src.cones_count_all =
      ter_dv_info_cones_count_all_encode(static_cast<float>(std::min<uint32_t>(msg->cones_count_total, 131071u)));
    src.lap_counter =
      ter_dv_info_lap_counter_encode(static_cast<float>(std::min<uint32_t>(msg->lap_counter, 15u)));
    // MissionInfo carries no mission type; report MANUAL.
    src.mission = TER_DV_INFO_MISSION_MANUAL_CHOICE;

    struct can_frame frame {};
    frame.can_id = TER_DV_INFO_FRAME_ID;  // standard frame, no flags
    frame.can_dlc = TER_DV_INFO_LENGTH;

    const int packed = ter_dv_info_pack(frame.data, &src, sizeof(frame.data));
    if (packed < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to pack DV_info");
      return;
    }

    if (write(can_fd_, &frame, sizeof(frame)) != static_cast<ssize_t>(sizeof(frame))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Failed to write DV_info to CAN: %s", std::strerror(errno));
    }
  }

  // --- CAN DV_system_status -> ROS AsInfo ---
  void rx_poll()
  {
    if (can_fd_ < 0) {
      return;
    }

    struct can_frame frame;
    while (true) {
      const ssize_t n = read(can_fd_, &frame, sizeof(frame));
      if (n < 0) {
        break;  // EAGAIN / EWOULDBLOCK: no more frames pending
      }
      if (n < static_cast<ssize_t>(sizeof(frame))) {
        continue;
      }
      if ((frame.can_id & CAN_SFF_MASK) != TER_DV_SYSTEM_STATUS_FRAME_ID) {
        continue;
      }

      struct ter_dv_system_status_t dst;
      if (ter_dv_system_status_unpack(&dst, frame.data, frame.can_dlc) != 0) {
        continue;
      }

      mirena_common::msg::AsInfo out;
      out.status = map_as_status(dst.as_status);
      out.mission = map_ami_mission(dst.ami_state);
      as_info_pub_->publish(out);
    }
  }

  // DBC AS_status choices use a different enumeration than the ROS AsInfo status constants.
  static uint8_t map_as_status(uint8_t dbc_value)
  {
    using Msg = mirena_common::msg::AsInfo;
    switch (dbc_value) {
      case TER_DV_SYSTEM_STATUS_AS_STATUS_AS_STATUS_OFF_CHOICE:       return Msg::AS_STATUS_OFF;
      case TER_DV_SYSTEM_STATUS_AS_STATUS_AS_STATUS_READY_CHOICE:     return Msg::AS_STATUS_READY;
      case TER_DV_SYSTEM_STATUS_AS_STATUS_AS_STATUS_DRIVING_CHOICE:   return Msg::AS_STATUS_DRIVING;
      case TER_DV_SYSTEM_STATUS_AS_STATUS_AS_STATUS_FINISHED_CHOICE:  return Msg::AS_STATUS_FINISHED;
      case TER_DV_SYSTEM_STATUS_AS_STATUS_AS_STATUS_EMERGENCY_CHOICE: return Msg::AS_STATUS_EMERGENCY;
      default:                                                        return Msg::AS_STATUS_OFF;
    }
  }

  // ROS MissionInfo status maps onto the DBC DV_info MISSION_STATUS choices.
  // DV_info has no FAILED choice; both terminal states (FINISHED/FAILED) map to
  // FINISHED, and any unknown value falls back to NONE.
  static uint8_t map_mission_status(uint8_t status)
  {
    using Msg = mirena_common::msg::MissionInfo;
    switch (status) {
      case Msg::MISSION_STATUS_INACTIVE:    return TER_DV_INFO_MISSION_STATUS_NONE_CHOICE;
      case Msg::MISSION_STATUS_CONFIGURING: return TER_DV_INFO_MISSION_STATUS_SELECTED_CHOICE;
      case Msg::MISSION_STATUS_READY:       return TER_DV_INFO_MISSION_STATUS_READY_CHOICE;
      case Msg::MISSION_STATUS_RUNNING:     return TER_DV_INFO_MISSION_STATUS_RUNNING_CHOICE;
      case Msg::MISSION_STATUS_FINISHED:    return TER_DV_INFO_MISSION_STATUS_FINISHED_CHOICE;
      case Msg::MISSION_STATUS_FAILED:      return TER_DV_INFO_MISSION_STATUS_FINISHED_CHOICE;
      default:                              return TER_DV_INFO_MISSION_STATUS_NONE_CHOICE;
    }
  }

  // DBC AMI_state choices map onto the ROS AsInfo mission constants.
  static uint8_t map_ami_mission(uint8_t ami_state)
  {
    using Msg = mirena_common::msg::AsInfo;
    switch (ami_state) {
      case TER_DV_SYSTEM_STATUS_AMI_STATE_AMI_STATE_ACCELERATION_CHOICE: return Msg::AS_MISSION_ACCELERATION;
      case TER_DV_SYSTEM_STATUS_AMI_STATE_AMI_STATE_SKIDPAD_CHOICE:      return Msg::AS_MISSION_SKIDPAD;
      case TER_DV_SYSTEM_STATUS_AMI_STATE_AMI_STATE_TRACKDRIVE_CHOICE:   return Msg::AS_MISSION_TRACKDRIVE;
      case TER_DV_SYSTEM_STATUS_AMI_STATE_AMI_STATE_BRAKETEST_CHOICE:    return Msg::AS_MISSION_BRAKE_TEST;
      case TER_DV_SYSTEM_STATUS_AMI_STATE_AMI_STATE_INSPECTION_CHOICE:   return Msg::AS_MISSION_INSPECTION;
      case TER_DV_SYSTEM_STATUS_AMI_STATE_AMI_STATE_AUTOCROSS_CHOICE:    return Msg::AS_MISSION_AUTOCROSS;
      default:                                                          return Msg::AS_MISSION_NONE;
    }
  }

  std::string iface_;
  int can_fd_ {-1};
  rclcpp::Subscription<mirena_common::msg::CarControl>::SharedPtr control_sub_;
  rclcpp::Subscription<mirena_common::msg::MissionInfo>::SharedPtr mission_info_sub_;
  rclcpp::Publisher<mirena_common::msg::AsInfo>::SharedPtr as_info_pub_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MirenaCan>());
  rclcpp::shutdown();
  return 0;
}
