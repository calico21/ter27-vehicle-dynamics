#pragma once

#include <rclcpp/rclcpp.hpp>

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

namespace mirena::IMU
{
    class Wrapper
    {

        rclcpp::Logger logger;

        bool imu_is_connected_ = false;
        VN::Sensor imu_;

    public:
        Wrapper(rclcpp::Logger log) :  logger(log) {} 

        auto get_next_measurement() {
            auto msg = imu_.getNextMeasurement();
            return msg;
        }

        /// Connect and configure, or leave the port closed. The caller decides
        /// what to do about failure by asking is_connected(); nothing has ever
        /// wanted the specific VN::Error, so it is not returned.
        void connect_and_config(const char *imu_dev_path,
                                const std::vector<double> &antenna_a_offset_,
                                const std::vector<double> &gnss_baseline_,
                                const std::vector<double> &gnss_baseline_unc_,
                                uint16_t rate_divisor)
        {
            RCLCPP_INFO(logger, "Connecting to IMU on %s ...", imu_dev_path);
            if (imu_.autoConnect(imu_dev_path) != VN::Error::None)
            {
                RCLCPP_ERROR(logger, "Could not open %s and/or no IMU found on it.", imu_dev_path);
                return;
            }

            // From here the port is open, so every failure below has to give it
            // back before returning. Leaving it held meant the next attempt
            // called autoConnect on a device this process already had.
            const auto write = [&](auto &reg, const char *what) {
                if (imu_.writeRegister(&reg) == VN::Error::None)
                {
                    return true;
                }
                RCLCPP_WARN(logger, "Could not write %s register; closing the port to retry.", what);
                imu_.disconnect();
                return false;
            };

            // --- Antenna A offset ---
            VN::Registers::GNSS::GnssAOffset antenna_a;
            antenna_a.positionX = static_cast<float>(antenna_a_offset_[0]);
            antenna_a.positionY = static_cast<float>(antenna_a_offset_[1]);
            antenna_a.positionZ = static_cast<float>(antenna_a_offset_[2]);
            if (!write(antenna_a, "GNSS antenna A offset"))
            {
                return;
            }

            // --- GNSS compass baseline (antenna B relative to A) ---
            VN::Registers::GNSSCompass::GnssCompassBaseline baseline;
            baseline.positionX = static_cast<float>(gnss_baseline_[0]);
            baseline.positionY = static_cast<float>(gnss_baseline_[1]);
            baseline.positionZ = static_cast<float>(gnss_baseline_[2]);
            baseline.uncertaintyX = static_cast<float>(gnss_baseline_unc_[0]);
            baseline.uncertaintyY = static_cast<float>(gnss_baseline_unc_[1]);
            baseline.uncertaintyZ = static_cast<float>(gnss_baseline_unc_[2]);
            if (!write(baseline, "GNSS compass baseline"))
            {
                return;
            }

            // --- Binary output 1: exactly the fields this driver consumes, at the
            //     configured rate. Requesting them explicitly guarantees the
            //     optionals read below are always present. ---
            VN::Registers::System::BinaryOutput1 bin_out_1;
            bin_out_1.asyncMode.emplace();
            bin_out_1.asyncMode->serial1 = true;
            bin_out_1.asyncMode->serial2 = false;
            bin_out_1.rateDivisor = rate_divisor;
            // Three timing fields, for three different jobs. timeStartup is
            // the sensor's own monotonic clock (MCU TCXO, +-20 ppm), stamped
            // at the sample -- it is what the published stamps are built on,
            // because it is free of every latency between the sample and this
            // process. timeGps is absolute UTC, kept for the host-clock skew
            // diagnostic and the legacy stamp mode. timeStatus says whether
            // the GPS one means anything yet.
            bin_out_1.time.timeStartup = true;
            bin_out_1.time.timeGps = true;
            bin_out_1.time.timeStatus = true;
            bin_out_1.imu.accel = true;
            bin_out_1.imu.angularRate = true;
            bin_out_1.attitude.quaternion = true;
            bin_out_1.ins.insStatus = true;
            bin_out_1.ins.posLla = true;
            bin_out_1.ins.velBody = true;
            bin_out_1.attitude.linBodyAcc = true;
            // Onboard-filter uncertainties (1 sigma) — used to fill the INS
            // odometry covariance published for the state estimator.
            bin_out_1.ins.posU = true;
            bin_out_1.ins.velU = true;
            bin_out_1.attitude.yprU = true;

            if (!write(bin_out_1, "binary output 1"))
            {
                return;
            }

            // Silence the other two async outputs, in case a previous session
            // or a saved configuration left them streaming into the same port.
            // A default-constructed AsyncMode is serial1 = serial2 = false;
            // rateDivisor is then irrelevant but must still be set, because
            // BinaryOutput::toString() emits nothing at all without it.
            VN::Registers::System::BinaryOutput2 bin_out_2;
            bin_out_2.asyncMode.emplace();
            bin_out_2.rateDivisor = rate_divisor;
            VN::Registers::System::BinaryOutput3 bin_out_3;
            bin_out_3.asyncMode.emplace();
            bin_out_3.rateDivisor = rate_divisor;
            if (!write(bin_out_2, "binary output 2") || !write(bin_out_3, "binary output 3"))
            {
                return;
            }

            imu_is_connected_ = true;
        }

        void disconnect()
        {
            imu_.disconnect();
            // The flag is the whole point of this call. Leaving it set meant
            // is_connected() answered yes for ever after the first connect, so
            // the reconnect branch in the publish loop was unreachable and one
            // dropped cable wedged the driver until it was restarted.
            imu_is_connected_ = false;
        }

        bool is_connected(){
            return imu_is_connected_;
        }
    };
}
