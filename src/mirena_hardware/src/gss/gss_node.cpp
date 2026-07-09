#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
#include <memory>
#include <utility> 

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

// --- Constantes ---
const double LENS_FOCAL_LENGTH_M = 0.0028;
const double SENSOR_NATIVE_PIXEL_SIZE_M = 3.0e-6;
const int SENSOR_NATIVE_WIDTH = 1280;
const int SENSOR_NATIVE_HEIGHT = 800;
const int CAMERA_ID = 0; 
const int FRAME_WIDTH = 640; 
const int FRAME_HEIGHT = 400;
const int FPS = 120;
const double DEFAULT_HEIGHT_M = 0.235;
const int ECC_MAX_ITER = 50;
const double ECC_EPSILON = 0.01;
const double MIN_CORRELATION = 0.8;
class gssNode : public rclcpp::Node
{
public:
    gssNode(): Node("gss_node")
    {
        this->declare_parameter("camera_id", 0);
        this->declare_parameter("camera_height_m", DEFAULT_HEIGHT_M);

        int camera_id = this->get_parameter("camera_id").as_int();
        double camera_height = this->get_parameter("camera_height_m").as_double();

        // 1. Configuración de parámetros de cámara y binning
        double binning_factor_x = (double)SENSOR_NATIVE_WIDTH / FRAME_WIDTH;
        double binning_factor_y = (double)SENSOR_NATIVE_HEIGHT / FRAME_HEIGHT;
        double binning_factor = (binning_factor_x + binning_factor_y) / 2.0;

        if (std::abs(binning_factor_x - binning_factor_y) > 0.1)
        {
            RCLCPP_WARN(this->get_logger(), "La relación de aspecto no coincide. Mediciones pueden ser incorrectas.");
        }

        double effective_pixel_size_m = SENSOR_NATIVE_PIXEL_SIZE_M * binning_factor;
        focal_length_px_ = LENS_FOCAL_LENGTH_M / effective_pixel_size_m;
        height_m_ = camera_height;

        // 2. Inicialización de la Cámara
        cap_.open(camera_id, cv::CAP_V4L2);
        if (!cap_.isOpened())
        {
            cap_.open(camera_id);
            if(!cap_.isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Error opening camera ID: %d", camera_id);
                throw std::runtime_error("Camera failed to open");
            }
        }
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
        cap_.set(cv::CAP_PROP_FPS, FPS);

        // 3. Captura del primer frame para inicializar
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty())
        {
            throw std::runtime_error("Error capturing initial frame");
        }

        cv::cvtColor(frame, prev_gray_, cv::COLOR_BGR2GRAY);
        cv::pyrDown(prev_gray_, prev_gray_small_);

        // 4. Inicialización de matrices ECC
        warp_ = cv::Mat::eye(2, 3, CV_32F);
        criteria_ = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, ECC_MAX_ITER, ECC_EPSILON);

        // 5. Crear Publisher y Timer
        publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/sensors/gss", 10);

        timer_ = this->create_wall_timer(33ms, std::bind(&gssNode::timer_callback, this));

        std::cout << "[INFO] Nodo iniciado. Tracking activo." << std::endl;
    }

private:
    // --- Lógica de Visión (Cálculo) ---
    std::pair<double, double> gssOdometria()
    {
        int64 t_frame_start = cv::getTickCount();

        cv::Mat frame, gray, gray_small;
        cap_ >> frame;

        if (frame.empty()) return {0.0, 0.0};

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::pyrDown(gray, gray_small);

        bool motion_estimated = false;
        double cc = 0.0;

        try
        {
            cc = cv::findTransformECC(prev_gray_small_, gray_small, warp_, cv::MOTION_EUCLIDEAN, criteria_);
            if (cc > MIN_CORRELATION)
            {
                motion_estimated = true;
            }
        }
        catch (cv::Exception &e)
        {
            return {0.0, 0.0};
        }

        float dx_pix = 0.0f, dy_pix = 0.0f, dtheta_deg = 0.0f;
        float dx_meters = 0.0f, dy_meters = 0.0f;
        
        double current_height_m = height_m_;

        if (motion_estimated)
        {
            dx_pix = warp_.at<float>(0, 2) * 2.0f;
            dy_pix = warp_.at<float>(1, 2) * 2.0f;
            dtheta_deg = std::atan2(warp_.at<float>(1, 0), warp_.at<float>(0, 0)) * 180.0 / CV_PI;

            dx_meters = (dx_pix * current_height_m) / focal_length_px_;
            dy_meters = (dy_pix * current_height_m) / focal_length_px_;

            // Acumulación de posición (Odometría Global)
            pose_angle_deg_ += dtheta_deg;
            float pose_angle_rad = pose_angle_deg_ * CV_PI / 180.0f;
            float cosA = std::cos(pose_angle_rad);
            float sinA = std::sin(pose_angle_rad);

            // Transformación de coordenadas: Del frame del robot al frame global
            // [Image of 2D mobile robot coordinate frame transformation]
            
            cv::Point2f delta_meters(dx_meters, dy_meters);
            cv::Point2f rotated_delta_meters(delta_meters.x * cosA - delta_meters.y * sinA,
                                             delta_meters.x * sinA + delta_meters.y * cosA);

            pose_pos_meters_ += rotated_delta_meters;
            distancia_total_x_ += dx_meters;
            distancia_total_y_ += dy_meters;
        }

        // Cálculo de Tiempo y FPS
        int64 t_frame_end = cv::getTickCount();
        double frame_time_sec = (t_frame_end - t_frame_start) / cv::getTickFrequency();
        double current_fps = (frame_time_sec > 0) ? (1.0 / frame_time_sec) : 0.0;

        // Velocidades Instantáneas (en el frame del robot)
        double vx_meters = dx_meters * current_fps;
        double vy_meters = dy_meters * current_fps;

        prev_gray_small_ = gray_small.clone();

        return {vx_meters, vy_meters};
    }

    // --- Callback del Timer (Publicación) ---
    void timer_callback()
    {
        // 1. Llamar a la lógica matemática
        std::pair<double, double> velocidades = gssOdometria();

        // 2. Publicar mensaje ROS
        auto message = geometry_msgs::msg::TwistStamped();
        message.header.stamp = this->get_clock()->now();
        message.header.frame_id = "base_link";

        message.twist.linear.x = velocidades.first;  
        message.twist.linear.y = velocidades.second; 
        message.twist.angular.z = 0.0; // Aquí podrías añadir dtheta_deg * fps si lo deseas

        publisher_->publish(message);
    }

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    cv::VideoCapture cap_;
    cv::Mat prev_gray_;
    cv::Mat prev_gray_small_;
    cv::Mat warp_;
    cv::TermCriteria criteria_;

    double focal_length_px_;
    double height_m_;

    // Variables de Estado
    cv::Point2f pose_pos_meters_{0.0f, 0.0f};
    float pose_angle_deg_ = 0.0f;
    double distancia_total_x_ = 0.0;
    double distancia_total_y_ = 0.0;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<gssNode>());
    } catch (const std::exception &e) {
        std::cerr << "Excepción en nodo: " << e.what() << std::endl;
    }
    rclcpp::shutdown();
    return 0;
}