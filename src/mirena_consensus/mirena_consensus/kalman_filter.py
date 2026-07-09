import casadi as ca

class KalmanFilter:
    def __init__(self, name_: str, x_size_: int, u_size_: int, z_size_: int, f_: ca.Function, h_: ca.Function):
        """
        Kalman filter where:
        - x_size is the size of the state vector: X
        - u_size is the size of the control vector: U
        - z_size is the size of the measure vector: Z
        - f is the state prediction function: X_{k|k-1} = f(X_{k-1|k-1}, U_{k-1}). Its signature is f: X, U -> X 
        - h is the measure prediction function: Zestimado_{k} = h(X_{k|k-1}). Its signature is f: X -> Z 
        """
        self.name = name_

        X_km1_km1 = ca.MX.sym("X_km1_km1", x_size_) # X_{k-1|k-1}
        P_km1_km1 = ca.MX.sym("P_km1_km1", x_size_, x_size_)  # P_{k-1|k-1}
        U = ca.MX.sym("U", u_size_)
        Z = ca.MX.sym("Z", z_size_)

        Q_km1 = ca.MX.sym("Q_km1", x_size_, x_size_) # Process noise
        R_k = ca.MX.sym("R_km1", z_size_, z_size_) # Observation noise

        F_func, H_func = self.generate_jacobians(x_size_, u_size_, f_, h_)
        I = ca.DM_eye(x_size_)

        # This shit down here is literally almost copied word for word from wikipedia GODDD i love casadi in the most unironical way possible
        F_k = F_func(X_km1_km1, U)
        # Predict 
        X_k_km1 = f_(X_km1_km1, U) # X_{k|k-1}
        P_k_km1 = F_k @ P_km1_km1 @ ca.transpose(F_k) + Q_km1 # P_{k|k-1}

        H_k = H_func(X_k_km1)
        # Update
        Y_diff = Z - h_(X_k_km1)
        S_k = H_k @ P_k_km1 @ ca.transpose(H_k) + R_k
        K_k = P_k_km1 @ ca.transpose(H_k) @ ca.solve(S_k, ca.DM_eye(z_size_))
        X_k_k = X_k_km1 + K_k @ Y_diff # X_{k|k}
        P_k_k = (I - K_k @ H_k) @ P_k_km1  # P_{k|k}


        self.predict = ca.Function(f"{self.name}_predict", [X_km1_km1, P_km1_km1, U, Q_km1], [X_k_km1, P_k_km1])
        self.update = ca.Function(f"{self.name}_update", [X_km1_km1, P_km1_km1, U, Z, Q_km1, R_k], [X_k_k, P_k_k])


    @staticmethod
    def generate_jacobians(x_size: int, u_size: int, f: ca.Function, h: ca.Function):
        dummy_x = ca.MX.sym("dummy_x", x_size)
        dummy_u = ca.MX.sym("dummy_u", u_size)

        F_jacobian = ca.Function('F', [dummy_x, dummy_u], [ca.jacobian(f(dummy_x, dummy_u), dummy_x)])
        H_jacobian = ca.Function('H', [dummy_x], [ca.jacobian(h(dummy_x), dummy_x)])

        return F_jacobian, H_jacobian
    

    def generate_c(self, dir: str):
        codegen = ca.CodeGenerator(f"{self.name}.c", {"with_header": True, "prefix": self.name, "main": False})
        codegen.add(self.update)
        codegen.add(self.predict)
        codegen.generate(dir)