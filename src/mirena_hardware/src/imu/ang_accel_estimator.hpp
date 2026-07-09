#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tf2/LinearMath/Vector3.hpp>

namespace mirena::AngAccelEstimator {
class SavitzkyGolay
{
private:
    int window_size;

public:
    explicit SavitzkyGolay(int window_size) : window_size(window_size)
    {
        if (window_size % 2 == 0)
        {
            window_size += 1;
        }
        if (window_size < 3)
        {
            throw std::invalid_argument("Window size must be at least 3");
        }
    }

    // Derivative with variable time steps
    double derivative(const std::vector<double> &data,
                      const std::vector<double> &time,
                      int center_index) const
    {
        int n = window_size;
        int m = (n - 1) / 2;

        // Check bounds
        if (center_index - m < 0 || center_index + m >= static_cast<int>(data.size()))
        {
            throw std::out_of_range("Center index too close to boundary");
        }
        if (data.size() != time.size())
        {
            throw std::invalid_argument("Data and time vectors must have same size");
        }

        double t0 = time[center_index];

        // Build Vandermonde matrix for linear fit
        Eigen::MatrixXd X(n, 2);
        for (int i = 0; i < n; ++i)
        {
            double dt = time[center_index - m + i] - t0;
            X(i, 0) = 1.0;
            X(i, 1) = dt;
        }

        // Solve least squares: more stable than direct inverse
        Eigen::MatrixXd coeffs = X.colPivHouseholderQr().solve(
            Eigen::MatrixXd::Identity(n, n));

        // Extract window
        Eigen::VectorXd window(n);
        for (int i = 0; i < n; ++i)
        {
            window(i) = data[center_index - m + i];
        }

        // Derivative is the second row (index 1) of coeffs
        return coeffs.row(1).dot(window);
    }
};

class AngAccelEstimator
{
private:
    const size_t buffer_size;
    int write_index;
    int num_samples_stored;

    // Circular buffers
    std::vector<double> w_x_buffer;
    std::vector<double> w_y_buffer;
    std::vector<double> w_z_buffer;
    std::vector<double> time_buffer;

    SavitzkyGolay sg_x;
    SavitzkyGolay sg_y;
    SavitzkyGolay sg_z;

    // Get the center index
    int get_center_index() const
    {
        if (num_samples_stored < static_cast<int>(buffer_size))
        {
            return -1;
        }
        int m = static_cast<int>(buffer_size - 1) / 2;
        int center = write_index - 1 - m;
        if (center < 0)
        {
            center += buffer_size;
        }
        return center;
    }

    // Get element from circular buffer with offset from center
    double get_element(const std::vector<double> &buffer, int center_idx, int offset) const
    {
        int idx = center_idx + offset;
        // Wrap around if needed
        if (idx < 0)
            idx += buffer_size;
        else if (idx >= static_cast<int>(buffer_size))
            idx -= buffer_size;
        return buffer[idx];
    }

public:
    explicit AngAccelEstimator(size_t window_size)
        : buffer_size(window_size),
          write_index(0),
          num_samples_stored(0),
          w_x_buffer(window_size, 0.0),
          w_y_buffer(window_size, 0.0),
          w_z_buffer(window_size, 0.0),
          time_buffer(window_size, 0.0),
          sg_x(static_cast<int>(window_size)),
          sg_y(static_cast<int>(window_size)),
          sg_z(static_cast<int>(window_size))
    {
    }

    tf2::Vector3 estimate_ang_accel(tf2::Vector3 ang_vel, double time)
    {
        // Write to circular buffers
        w_x_buffer[write_index] = ang_vel[0];
        w_y_buffer[write_index] = ang_vel[1];
        w_z_buffer[write_index] = ang_vel[2];
        time_buffer[write_index] = time;

        write_index = (write_index + 1) % buffer_size;

        if (num_samples_stored < static_cast<int>(buffer_size))
        {
            num_samples_stored++;
            return tf2::Vector3(0.0, 0.0, 0.0);
        }

        int center_idx = get_center_index();
        int m = static_cast<int>(buffer_size - 1) / 2;

        // Build X matrix once
        Eigen::MatrixXd X(buffer_size, 2);
        double t0 = time_buffer[center_idx];
        for (int i = 0; i < static_cast<int>(buffer_size); ++i)
        {
            int idx = center_idx - m + i;
            if (idx < 0)
                idx += buffer_size;
            else if (idx >= static_cast<int>(buffer_size))
                idx -= buffer_size;

            X(i, 0) = 1.0;
            X(i, 1) = time_buffer[idx] - t0;
        }

        // Precompute coefficients once
        Eigen::MatrixXd coeffs = X.colPivHouseholderQr().solve(
            Eigen::MatrixXd::Identity(buffer_size, buffer_size));

        // Build windows for all three axes in one loop
        Eigen::VectorXd window_x(buffer_size);
        Eigen::VectorXd window_y(buffer_size);
        Eigen::VectorXd window_z(buffer_size);

        for (int i = 0; i < static_cast<int>(buffer_size); ++i)
        {
            int idx = center_idx - m + i;
            if (idx < 0)
                idx += buffer_size;
            else if (idx >= static_cast<int>(buffer_size))
                idx -= buffer_size;

            window_x(i) = w_x_buffer[idx];
            window_y(i) = w_y_buffer[idx];
            window_z(i) = w_z_buffer[idx];
        }

        // Use the same coefficients for all three axes
        Eigen::RowVectorXd deriv_coeffs = coeffs.row(1);
        double alpha_x = deriv_coeffs.dot(window_x);
        double alpha_y = deriv_coeffs.dot(window_y);
        double alpha_z = deriv_coeffs.dot(window_z);

        return tf2::Vector3(alpha_x, alpha_y, alpha_z);
    }
};
}