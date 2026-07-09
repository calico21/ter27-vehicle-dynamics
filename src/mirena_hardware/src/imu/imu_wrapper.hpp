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

        VN::Error connect_and_config(const char *imu_dev_path, 
            std::vector<double> antenna_a_offset_,
            std::vector<double> gnss_baseline_,
            std::vector<double> gnss_baseline_unc_, 
        uint16_t rate_divisor)
        {
            RCLCPP_INFO(logger, "Connecting to IMU on %s ...", imu_dev_path);
            VN::Error err = imu_.autoConnect(imu_dev_path);
            if (err != VN::Error::None)
            {
                RCLCPP_ERROR(logger, "Could not open %s and/or no IMU found on it.", imu_dev_path);
                return err;
            }

            // --- Antenna A offset ---
            VN::Registers::GNSS::GnssAOffset antenna_a;
            antenna_a.positionX = static_cast<float>(antenna_a_offset_[0]);
            antenna_a.positionY = static_cast<float>(antenna_a_offset_[1]);
            antenna_a.positionZ = static_cast<float>(antenna_a_offset_[2]);
            if ((err = imu_.writeRegister(&antenna_a)) != VN::Error::None)
            {
                RCLCPP_WARN(logger, "Could not write GNSS antenna A offset register.");
                return err;
            }

            // --- GNSS compass baseline (antenna B relative to A) ---
            VN::Registers::GNSSCompass::GnssCompassBaseline baseline;
            baseline.positionX = static_cast<float>(gnss_baseline_[0]);
            baseline.positionY = static_cast<float>(gnss_baseline_[1]);
            baseline.positionZ = static_cast<float>(gnss_baseline_[2]);
            baseline.uncertaintyX = static_cast<float>(gnss_baseline_unc_[0]);
            baseline.uncertaintyY = static_cast<float>(gnss_baseline_unc_[1]);
            baseline.uncertaintyZ = static_cast<float>(gnss_baseline_unc_[2]);
            if ((err = imu_.writeRegister(&baseline)) != VN::Error::None)
            {
                RCLCPP_WARN(logger, "Could not write GNSS compass baseline register.");
                return err;
            }

            // --- Binary output 1: exactly the fields this driver consumes, at the
            //     configured rate. Requesting them explicitly guarantees the
            //     optionals read below are always present. ---
            VN::Registers::System::BinaryOutput1 bin_out_1;
            bin_out_1.asyncMode.emplace();
            bin_out_1.asyncMode->serial1 = true;
            bin_out_1.asyncMode->serial2 = false;
            bin_out_1.rateDivisor = rate_divisor;
            bin_out_1.time.timeGps = true;
            bin_out_1.time.timeStatus = true;
            bin_out_1.imu.accel = true;
            bin_out_1.imu.angularRate = true;
            bin_out_1.attitude.quaternion = true;
            bin_out_1.ins.insStatus = true;
            bin_out_1.ins.posLla = true;
            bin_out_1.ins.velBody = true;
            bin_out_1.attitude.linBodyAcc = true;
            bin_out_1.gnss.gnss1SatInfo=true;
            bin_out_1.gnss2.gnss2SatInfo=true;

            if ((err = imu_.writeRegister(&bin_out_1)) != VN::Error::None)
            {
                RCLCPP_WARN(logger, "Could not write binary output 1 register.");
                return err;
            }

            VN::Registers::System::BinaryOutput2 bin_out_2;
            bin_out_2.asyncMode.emplace();
            bin_out_2.asyncMode->serial1 = false;
            bin_out_2.asyncMode->serial2 = false;
            bin_out_2.rateDivisor = rate_divisor;
            if ((err = imu_.writeRegister(&bin_out_2)) != VN::Error::None)
            {
                RCLCPP_WARN(logger, "Could not write binary output 2 register.");
                return err;
            }

            VN::Registers::System::BinaryOutput3 bin_out_3;
            bin_out_3.asyncMode.emplace();
            bin_out_3.asyncMode->serial1 = false;
            bin_out_3.asyncMode->serial2 = false;
            bin_out_3.rateDivisor = rate_divisor;
            if ((err = imu_.writeRegister(&bin_out_3)) != VN::Error::None)
            {
                RCLCPP_WARN(logger, "Could not write binary output 3 register.");
                return err;
            }

            imu_is_connected_ = true;
            return VN::Error::None;
        }

        VN::Error write_config()
        {
            VN::Error err;
            if ((err = imu_.writeSettings()) != VN::Error::None)
            {
                RCLCPP_WARN(logger, "Could not persist config to flash.");
                return err;
            }
            RCLCPP_INFO(logger, "Sensor config persisted to non-volatile flash.");
        }

        VN::Error print_imu_info()
        {
            // Read back the geometry for confirmation.
            VN::Registers::GNSS::GnssAOffset a_rd;
            VN::Registers::GNSSCompass::GnssCompassBaseline b_rd;
            if (imu_.readRegister(&a_rd) == VN::Error::None && imu_.readRegister(&b_rd) == VN::Error::None)
            {
                RCLCPP_INFO(logger, "GNSS antenna A offset: %f, %f, %f",
                            a_rd.positionX.value(), a_rd.positionY.value(), a_rd.positionZ.value());
                RCLCPP_INFO(logger, "GNSS compass baseline: %f, %f, %f (unc %f, %f, %f)",
                            b_rd.positionX.value(), b_rd.positionY.value(), b_rd.positionZ.value(),
                            b_rd.uncertaintyX.value(), b_rd.uncertaintyY.value(), b_rd.uncertaintyZ.value());
            }

            return VN::Error::None;
        }

        VN::Error disconnect()
        {
            imu_.disconnect();
            return VN::Error::None;
        }

        bool is_connected(){
            return imu_is_connected_;
        }
    };
}
