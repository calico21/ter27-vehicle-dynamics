from .sqrt_extended_kalman_filter import SqrtExtendedKalmanFilter
from .car_model import BlendedBicycleSystemDynamics, SymCarParameters
import casadi as ca

X_SIZE = 6
U_SIZE = 3
Z_GPS_SIZE = 2
Z_IMU_SIZE = 2
Z_GSS_SIZE = 2

class Filters:

    
    def __init__(self):
        #Fordwar euler for simplicitys sake idfk
        X = ca.MX.sym("", X_SIZE)
        U = ca.MX.sym("", U_SIZE)
        dt = U[2]

        f_dot = BlendedBicycleSystemDynamics().get_system_dynamics()
        default_params = SymCarParameters.get_default_parameters_getter()(0)

        f = ca.Function("f", [X, U], [X + dt * f_dot(X, U[0:2], default_params)])
        h_gps = ca.Function("h_gps", [X], [ca.vertcat(X[0], X[1])]) # Z IS [x, y] where x and y are on the global frame
        h_imu = ca.Function("h_imu", [X], [ca.vertcat(X[2], X[5])]) # Z IS [\psi, \omega] where \psi is in the global frame
        h_gss = ca.Function("h_gss", [X], [ca.vertcat(X[3], X[4])]) # Z is [u, v] where u and v are in the car frame (longitudinal and lateral velocity)

        self.gps_kalman = SqrtExtendedKalmanFilter("gps_kalman", X_SIZE, U_SIZE, Z_GPS_SIZE, f, h_gps)
        self.imu_kalman = SqrtExtendedKalmanFilter("imu_kalman", X_SIZE, U_SIZE, Z_IMU_SIZE, f, h_imu)
        self.gss_kalman = SqrtExtendedKalmanFilter("gss_kalman", X_SIZE, U_SIZE, Z_GSS_SIZE, f, h_gss)