import casadi as ca
import numpy as np

class SymCarParameters:

    class Defaults():
        """
        Car Parameters
        ----------
        m [kg] : Total vehicle mass. 
        Iz[kg*m²] : Yaw moment of inertia
        Lf [m]: Distance from center of gravity (CG) to front axle.
        Lr [m]: Distance from center of gravity (CG) to rear axle.     
        """
        m: float = 250      # kg - Total vehicle mass
        Iz: float = 110     # kg*m² - Yaw moment of inertia  
        Lf: float = 0.8     # m - Distance from CG to front axle
        Lr: float = 0.75      # m - Distance from CG to rear axle
        kf: float = 35000    # N/rad - Front axle cornering stiffness
        kr: float = 35000    # N/rad - Rear axle cornering stiffness

        C_M = 2200     # Coeficiente de comanda
        C_R0 = 0        # Coeficiente de fricción
        C_R2 = 0        # Coeficiente cuadratico de drag

        B = 9
        C = 1.4
        D = 2.33

        P_LONG = 2
        P_ELLIPSE = 4

    N_OF_PARAMS = 14

    def __init__(self, name: str):
        self._params = ca.MX.sym(name, SymCarParameters.N_OF_PARAMS)

    def get_sym(self) -> ca.MX:
        return self._params

    @property
    def m(self):
        return self._params[0]
    
    @property
    def Iz(self):
        return self._params[1]
    
    @property
    def Lf(self):
        return self._params[2]
    
    @property
    def Lr(self):
        return self._params[3]
    
    @property
    def kf(self):
        return self._params[4]
    
    @property
    def kr(self):
        return self._params[5]
    
    @property
    def C_M(self):
        return self._params[6]
    
    @property
    def C_R0(self):
        return self._params[7]
    
    @property
    def C_R2(self):
        return self._params[8]
    
    @property
    def B(self):
        return self._params[9]
    
    @property
    def C(self):
        return self._params[10]
    
    @property
    def D(self):
        return self._params[11]
    
    @property
    def P_LONG(self):
        return self._params[12]

    @property
    def P_ELLIPSE(self):
        return self._params[13]

    @staticmethod
    def get_default_parameters_getter() -> ca.Function:
        """
        Returns a casadi function that takes no parameters and returns a symbolic matrix of default car parameters
        Done this way so that this shit can be exported to C as a casadi function lmao this is bad design
        Use:
        param_getter = SymCarParameters.get_default_parameters_getter()
        params_sym: ca.DM = param_getter(0)
        El 0 es dummy porque la interfaz de casadi es estupida y no te deja crear funciones sin argumentos de forma comoda ugh 
        """
        return ca.Function('CarParameters_get_defaults', [ca.MX.sym("dummy")], [ca.vertcat(
            SymCarParameters.Defaults.m,
            SymCarParameters.Defaults.Iz,
            SymCarParameters.Defaults.Lf,
            SymCarParameters.Defaults.Lr,
            SymCarParameters.Defaults.kf,
            SymCarParameters.Defaults.kr,
            SymCarParameters.Defaults.C_M,
            SymCarParameters.Defaults.C_R0,
            SymCarParameters.Defaults.C_R2,
            SymCarParameters.Defaults.B,
            SymCarParameters.Defaults.C,
            SymCarParameters.Defaults.D,
            SymCarParameters.Defaults.P_LONG,
            SymCarParameters.Defaults.P_ELLIPSE
        )])

class ASystemDynamics:
    def __init__(self):
        raise AssertionError("Called method on abstract class")
    
    def get_model_name(self) -> str:
        raise AssertionError("Called method on abstract class")
 
    def get_system_dynamics(self) -> ca.Function:
        raise AssertionError("Called method on abstract class")

class BlendedBicycleSystemDynamics(ASystemDynamics):
    def __init__(self):
        # ----------------------
        # State & Control
        # ----------------------
        
        # Vector  extendido de 6 elementos: [x, y, phi, u, v, omega]
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        phi = ca.MX.sym('phi')
        u = ca.MX.sym('u')
        v = ca.MX.sym('v')
        omega = ca.MX.sym('omega')

        X = ca.vertcat(
            x,
            y,
            phi,
            u,
            v,
            omega
        )  # Estado kinematico

        # Vector de control
        g = ca.MX.sym('g')  # Control de gas
        d = ca.MX.sym('delta')  # Volante

        U = ca.vertcat(
            g,  # Gas
            d  # Steer angle
        )

        _param = SymCarParameters("P")
        
        # ----------------------
        # Continous-time dynamics: X_dot = dX/dt
        # ----------------------
        # Vertical force
        K_RELAX = 20
        F_Z = -(_param.m * 9.81)/2


        # Modelo Fuerza en x del powertrain
        # Incluye drag inicial y drag cuadratico por aero
        f_x = _param.C_M * g - _param.C_R0 - _param.C_R2 * u**2

        epsilon = 0.001

        # Definición de slip angles
        # ca.fmax(u, 0.1) prevents division by zero at low speeds
        alpha_r = ca.arctan((v - _param.Lr*omega)/ca.sqrt(u**2+epsilon**2))  # Rear slip angle
        alpha_f = ca.arctan((v + _param.Lf*omega)/ca.sqrt(u**2+epsilon**2) - d)  # Front slip angle

        # Definición de fuerzas en y de rueda (Lateral forces using Pacejka simplified model)
        f_r_y = F_Z*_param.D*ca.sin(_param.C*ca.atan(_param.B*alpha_r))
        f_f_y = F_Z*_param.D*ca.sin(_param.C*ca.atan(_param.B*alpha_f))

        # Lat speed target
        omega_t = u*d/(_param.Lf+_param.Lr)

        # Modelo cinemático
        kin_X_dot = ca.Function('KinDerivative',
                                    [X, U, _param.get_sym()],
                                    [ca.vertcat(
                                        u*ca.cos(phi) - v*ca.sin(phi),
                                        u*ca.sin(phi) + v*ca.cos(phi),
                                        omega,
                                        f_x/_param.m - 0.04*u*u,
                                        K_RELAX*-v, # Decays exiting lateral speed
                                        K_RELAX*(omega_t-omega) #  Drives to target rate
                                    )])

        # Modelo dinámico
        dyn_X_dot = ca.Function('DynDerivative',
                                    [X, U, _param.get_sym()],
                                    [ca.vertcat(
                                        u*ca.cos(phi) - v*ca.sin(phi), 
                                        u*ca.sin(phi) + v*ca.cos(phi),
                                        omega,
                                        (1/_param.m)*(f_x - f_f_y*ca.sin(d) + _param.m*v*omega) - 0.04*u*u,
                                        (1/_param.m)*(f_r_y + f_f_y*ca.cos(d) - _param.m*u*omega),
                                        (1/_param.Iz)*(f_f_y*_param.Lf*ca.cos(d) - f_r_y*_param.Lr)
                                    )])
        # Modelo blending (Soft)
        #blend = ca.fmin(ca.fmax((ca.fabs(u)-U_BLEND_MIN)/(U_BLEND_MAX-U_BLEND_MIN),0),1)
        K_SMOOTH = 2
        U_TRANS = 3
        blend = 0.5 * (1 + ca.tanh(K_SMOOTH * (ca.fabs(u) - U_TRANS)))
        blend_X_dot = ca.Function('BlendDerivative',
                                    [X, U, _param.get_sym()],
                                    [blend*dyn_X_dot(X,U, _param.get_sym()) + (1-blend)*kin_X_dot(X,U, _param.get_sym())])

        self._system_dynamics = blend_X_dot
        
    def get_model_name(self) -> str:
        return "BlendedBicycle"
    
    def get_system_dynamics(self) -> ca.Function:
        """
        Get system dynamics: X_dot(X, U), where:
        X = [x, y, phi, u, v, omega]
        U = [ax, delta] 
        """
        return self._system_dynamics
