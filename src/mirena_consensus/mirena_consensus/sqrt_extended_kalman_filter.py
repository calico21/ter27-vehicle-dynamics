import casadi as ca

class SqrtExtendedKalmanFilter:
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
        sqrt_P_km1_km1 = ca.MX.sym("P_km1_km1", x_size_, x_size_)  # P_{k-1|k-1}
        U = ca.MX.sym("U", u_size_)
        Z = ca.MX.sym("Z", z_size_)

        sqrt_Q_km1 = ca.MX.sym("Q_km1", x_size_, x_size_) # Matrix sqrt of Process noise
        sqrt_R_k = ca.MX.sym("R_km1", z_size_, z_size_) # Matrix sqrt of Observation noise

        F_func, H_func = self.generate_jacobians(x_size_, u_size_, f_, h_)

        # This shit down here is literally almost copied word for word from a NASA paper GODDD i love casadi in the most unironical way possible
        F_k = F_func(X_km1_km1, U)
        # Predict 
        X_k_km1 = f_(X_km1_km1, U) # X_{k|k-1}
        A2T = SqrtExtendedKalmanFilter.build_A2T(x_size_, sqrt_Q_km1, F_k, sqrt_P_km1_km1)
        B2T, _ =  SqrtExtendedKalmanFilter.lq_factorization(A2T)
        sqrt_P_k_km1 = SqrtExtendedKalmanFilter.extract_from_B2T(x_size_, B2T)

        H_k = H_func(X_k_km1)
        # Update
        A1T = SqrtExtendedKalmanFilter.build_A1T(x_size_, z_size_, sqrt_R_k, H_k, sqrt_P_k_km1)
        B1T, _ = SqrtExtendedKalmanFilter.lq_factorization(A1T)
        sqrt_P_k_k, K_x_sqrt_Y_k , sqrt_Y_k = SqrtExtendedKalmanFilter.extract_from_B1T(x_size_, z_size_, B1T)
        sqrt_Y_k_inv = SqrtExtendedKalmanFilter.invert_lower_triangular(sqrt_Y_k)
        X_k_k = X_k_km1 + K_x_sqrt_Y_k @ sqrt_Y_k_inv @ (Z - h_(X_k_km1)) # X_{k|k}


        self.predict = ca.Function(f"{self.name}_predict", [X_km1_km1, sqrt_P_km1_km1, U, sqrt_Q_km1], [X_k_km1, sqrt_P_k_km1])
        self.update = ca.Function(f"{self.name}_update", [X_km1_km1, sqrt_P_km1_km1, U, Z, sqrt_Q_km1, sqrt_R_k], [X_k_k, sqrt_P_k_k])


    @staticmethod
    def generate_jacobians(x_size: int, u_size: int, f: ca.Function, h: ca.Function):
        dummy_x = ca.MX.sym("dummy_x", x_size)
        dummy_u = ca.MX.sym("dummy_u", u_size)

        F_jacobian = ca.Function('F', [dummy_x, dummy_u], [ca.jacobian(f(dummy_x, dummy_u), dummy_x)])
        H_jacobian = ca.Function('H', [dummy_x], [ca.jacobian(h(dummy_x), dummy_x)])

        return F_jacobian, H_jacobian
    
    @staticmethod
    def build_A1T(size_x: int, size_z: int, sqrt_R_k: ca.MX, H_k: ca.MX, sqrt_P_k_km1: ca.MX) -> ca.MX:
        zeros = ca.MX.zeros(size_x, size_z)
        return ca.blockcat([
            [sqrt_R_k, H_k @ sqrt_P_k_km1], 
            [zeros, sqrt_P_k_km1]
            ])

    @staticmethod
    def extract_from_B1T(size_x: int, size_z: int, B1T: ca.MX) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Returns, in order: ( sqrt_P_k_k, K_x_sqrt_Y_k , sqrt_Y_k )"""
        blocks = ca.blocksplit(B1T, 
                               [0, size_z, size_x+size_z], 
                               [0, size_z, size_x+size_z]
                               )
        sqrt_Y_k = blocks[0][0]
        K_x_sqrt_Y_k = blocks[1][0]
        sqrt_P_k_k = blocks[1][1]
        return sqrt_P_k_k, K_x_sqrt_Y_k, sqrt_Y_k

    @staticmethod
    def build_A2T(size_x: int, sqrt_Q_km1: ca.MX, F_k: ca.MX, sqrt_P_km1_km1: ca.MX) -> ca.MX:
        zeros = ca.MX.zeros(size_x, size_x)
        return ca.blockcat([
            [sqrt_Q_km1, F_k @ sqrt_P_km1_km1], 
            [zeros, sqrt_P_km1_km1]
            ])

    @staticmethod
    def extract_from_B2T(size_x: int, B2T: ca.MX) -> ca.MX:
        """Returns: (sqrt_P_k_km1)"""
        blocks = ca.blocksplit(B2T, 
                               [0, size_x, size_x+size_x], 
                               [0, size_x, size_x+size_x]
                               )
        sqrt_P_k_km1 = blocks[0][0]
        return sqrt_P_k_km1

    @staticmethod
    def lq_factorization(A: ca.MX) -> tuple[ca.MX, ca.MX]:
        """Factorize A into LQ"""
        n, m = A.shape
        A_sx = ca.SX.sym('A', n, m)
        Q_sx, R_sx = ca.qr(A_sx)
        qr_function = ca.Function('qr_func', [A_sx], [Q_sx, R_sx])
        Q, R = qr_function(ca.transpose(A))
        ca.qr
        return ca.transpose(R), ca.transpose(Q)

    @staticmethod
    def invert_lower_triangular(L: ca.MX) -> ca.MX:
        """Stable inversion of lower triangular matrix using fordward substitution"""
        n = L.shape[0]
        L_inv = ca.MX.zeros(n, n)
        
        for j in range(n):
            # Solve for j-th column of inverse
            e = ca.MX.zeros(n, 1)
            e[j] = 1
            # Forward substitution
            for i in range(j, n):
                if i == j:
                    L_inv[i,j] = 1 / L[i,i]
                else:
                    L_inv[i,j] = -ca.mtimes(L[i,j:i], L_inv[j:i,j]) / L[i,i]
        
        return L_inv

    def generate_c(self, dir: str):
        codegen = ca.CodeGenerator(f"{self.name}.c", {"with_header": True, "prefix": self.name, "main": False})
        codegen.add(self.update)
        codegen.add(self.predict)
        codegen.generate(dir)