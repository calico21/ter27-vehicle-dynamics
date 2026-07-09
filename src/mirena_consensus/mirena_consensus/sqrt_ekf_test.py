from mirena_consensus.mirena_consensus.kalman.filters import Filters, X_SIZE, U_SIZE, Z_GPS_SIZE, Z_GSS_SIZE, Z_IMU_SIZE
import casadi as ca
import numpy as np

if __name__ == "__main__":
    x = np.array([1, 2, 0.2, 10, 0.01, 0.2])
    u = np.array([0.5, 0.2, 0.01])
    
    sqrt_P = np.diag([1.0, 1.0, 0.3, 0.7, 0.7, 0.3]).astype(np.float64)
    
    sqrt_Q = np.diag([0.1, 0.1, 0.03, 0.3, 0.3, 0.03]).astype(np.float64)

    sqrt_R_gps = np.diag([0.3, 0.3]).astype(np.float64)
    sqrt_R_gps[1][0] = 0.01

    filters = Filters()
    
    x_pred, sqrt_P_pred = filters.gps_kalman.predict(x, sqrt_P, u, sqrt_Q)
    x_pred = np.array(x_pred).flatten()
    print(x_pred)
    print(sqrt_P_pred)
    print(sqrt_P_pred*ca.transpose(sqrt_P_pred))

    z = np.array([x_pred[0]+0.01, x_pred[1]+0.01])

    #sqrt_Y_k, sqrt_P_k_k = filters.gps_kalman.test(x, sqrt_P, u, z, sqrt_Q, sqrt_R_gps)
    #print(sqrt_Y_k)
    
    updated_x, updated_P = filters.gps_kalman.update(x, sqrt_P, u, z, sqrt_Q, sqrt_R_gps)
    print(updated_x)
    print(updated_P)
