import math
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel,AcadosSim, AcadosSimSolver, ACADOS_INFTY
import matplotlib.pyplot as plt
import mirena_control.mpc.dbm_model as dbm

RAD30 = 30.0 * math.pi / 180.0

class MPCController:
    def __init__(self, path_points=20, N=20, T=10.0):
        self.PATH_POINTS = path_points
        self.N = N
        self.T = T
        self.dt = T / N

        self._build_ocp()
        self._build_solver()


    def _build_ocp(self):
        
        # Variables de problema
        p = ca.MX.sym('p')  # Progreso
        vp = ca.MX.sym('vp')  # Velocidad de Progreso

        # Vector de estado de problema extendido de 10 elementos: [x, y, phi, u, v, omega, g, delta, p,vp]
        X_P = ca.vertcat(
            dbm.X_CAR,  # Car state
            dbm.U_CAR,  # Control State
            p,     # Progress state
            vp      # Progress Speed
        )  # Estado de problema Aumentado
        #Derivada de estado
        X_P_DOT = ca.MX.sym('X_P_DOT',X_P.shape[0])

        # Control del problema y su derivada (Trabajamos sobre las derivadas del control para garantizar suavidad en la función de coste)
        # (Evolución de control del problema)

        # Control de problema (Progreso y acción)
        vp_dot = ca.MX.sym('vp_dot')
        g_dot = ca.MX.sym('gas')
        d_dot = ca.MX.sym('steer_angle')
        U_P = ca.vertcat(g_dot,d_dot, vp_dot)  # [g_dot, d_dot, vp_dot]


        # Creamos la función dinamica del problema (Bicycle Model)
        F_P_dot = ca.Function('MPCDerivative',
                                [X_P, U_P],
                                [ca.vertcat(
                                    dbm.F_BLEND(dbm.X_CAR, dbm.U_CAR),
                                    g_dot, # g
                                    d_dot, # d
                                    vp,    # p
                                    vp_dot # vp
                                )])
        
        # Parámetros del spline (dimensión fija)
        path_pts_x = ca.MX.sym('ctrl_pts_x', self.PATH_POINTS)
        path_pts_y = ca.MX.sym('ctrl_pts_y', self.PATH_POINTS)
        path_length = ca.MX.sym('path_length')


        # Agrupación del vector de parametros
        P_GLOBAL = ca.vertcat(path_pts_x, path_pts_y,path_length)


        # Normalized and clamped path (Avoids solver errors when bad lookups)
        p_norm = ca.fmin(ca.fmax(p/path_length, 0.0), 1.0) 

        # Referencia de pista: Puntos de la cuadrícula para el splines (0 a 1)
        norm_grid = np.linspace(0, 1, self.PATH_POINTS)

        # Splines independientes para cada coordenada (p indexa la curva)
        path_x = ca.interpolant('path_x', 'bspline', [norm_grid])
        path_y = ca.interpolant('path_y', 'bspline', [norm_grid])

        # Creamos la interpolacion simbolica (Ejecutada ya en el punto de progreso simbolico p)
        x_ref = path_x(p_norm,path_pts_x)
        y_ref = path_y(p_norm,path_pts_y)

        # Derivadas para la tangente
        dx_dp = ca.jacobian(x_ref, p)
        dy_dp = ca.jacobian(y_ref, p)


        # Much more stable
        EPSILON_REG = 1e-4
        norm_factor = ca.sqrt(dx_dp**2 + dy_dp**2 + EPSILON_REG**2)

        # Defintion of atan2
        tx = dx_dp / norm_factor
        ty = dy_dp / norm_factor

        e_c = ty * (dbm.x - x_ref) - tx * (dbm.y - y_ref) # Seno es ty, Coseno es tx
        e_l = -tx * (dbm.x - x_ref) - ty * (dbm.y - y_ref) # Coseno es tx, Seno es ty
        # Proyecciones error de contorno (e_c) y error de lag (e_l)
       

        # Virtual Car Pursuit approach
        # El producto escalar da 0 si alineado 2 si desalineado
        e_psi = 1.0 - (ca.cos(dbm.phi) * tx + ca.sin(dbm.phi) * ty)
        
        e_d = (dbm.x - x_ref)**2 + (dbm.y - y_ref)**2 
        # Coste: Maximizamos el progreso (vp) y penalizamos errores y cambios de control.
        cost_expr = (
            3 * e_d +
            2 * e_psi +
            -0.08 * vp +              # (3) Maximizar progreso (MINIMIZAR -vp)
            0.005 * g_dot**2 +        # (4) Penalización a tasa de gas
            0.5 * d_dot**2 +        # (5) Penalización a tasa de volante
            0.001 * vp_dot**2         # (6) Penalización a tasa de vp
        )

        cost_fun = ca.Function('cost_fun', [X_P, U_P, P_GLOBAL], [cost_expr])


        #cost_fun_e = ca.Function('cost_fun_e', [X_P, U_P, P_GLOBAL], [cost_expr_e])

        # Non linear  constraints (Tire constraints)
        
        P_H = ca.vertcat(dbm.F_F_S, dbm.F_R_S)

        # OCP INSTANTIATION

        # Instanciación del modelo
        self.model = AcadosModel()
        self.model.name = 'mpc_problem'

        # Especificamos Control Estado, del problema
        self.model.x = X_P  # vector de estados
        self.model.u = U_P  # vector de controles
        self.model.p_global = P_GLOBAL  # parámetros
        self.model.f_expl_expr = F_P_dot(X_P, U_P)
        
        #Implicit support
        self.model.xdot = X_P_DOT
        self.model.f_impl_expr = X_P_DOT - F_P_dot(X_P, U_P)

        self.ocp = AcadosOcp()
        self.ocp.model = self.model
        self.ocp.solver_options.N_horizon = self.N
        self.ocp.solver_options.tf = self.T
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.hpipm_mode = 'SPEED'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.nlp_solver_type = 'SQP'
        #self.ocp.solver_options.print_level = 4
        self.ocp.solver_options.levenberg_marquardt = 0.01
        #self.ocp.solver_options.qp_tol = 0.1
        self.ocp.solver_options.qp_solver_iter_max = 50
        self.ocp.solver_options.nlp_solver_max_iter = 100
        self.ocp.solver_options.nlp_solver_step_length = 0.8
        self.ocp.solver_options.regularize_method = 'PROJECT'
        self.ocp.solver_options.tol = 1e-2
        #self.ocp.solver_options.cost_discretization = 'INTEGRATOR'
        self.ocp.cost.cost_type = 'EXTERNAL'

        # Enable Merit Backtracking
        self.ocp.solver_options.nlp_solver_globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.globalization_alpha_min = 0.05

        # Use a slightly more robust line search
        self.ocp.solver_options.globalization_line_search_use_sufficient_descent = 1

        self.model.cost_expr_ext_cost = cost_fun(X_P,U_P,P_GLOBAL)
        #self.model.cost_expr_ext_cost_e = cost_fun_e(X_P,U_P,P_GLOBAL)
        
        nx = X_P.shape[0]
        nu = U_P.shape[0]
        

        # ------------------- State constraints (para todos los pasos de tiempo) -------------------
        self.ocp.constraints.idxbx = np.array([3,4,5,6,7,8,9])  # Aplicar constraints a todos los estados menos posición y progreso

        self.ocp.constraints.lbx = np.array([
            0,          # u (must be positive for dynamics)
            -2,         # v
            -3,  # omega
            -1,         # g (gas/throttle command)
            -RAD30,     # delta (steering angle)
            0,          # Progress
            -33        # vp (progress velocity)
        ])
        
        
        self.ocp.constraints.ubx = np.array([
            33,         # u (max speed)
            2,          # v
            3,   # omega
            1,          # g (gas/throttle command)
            RAD30,      # delta (steering angle)
            ACADOS_INFTY,         # p progress
            33         # vp (progress velocity)
        ])


        # ------------------- Control constraints (rate of change de controles) -------------------#
        self.ocp.constraints.idxbu = np.arange(nu)
        # [g_dot, delta_dot, vp_dot]
        self.ocp.constraints.lbu = np.array([
            -5,       # g_dot mínimo
            -5*RAD30,  # delta_dot mínimo
            -5       # vp_dot mínimo
        ])

        self.ocp.constraints.ubu = np.array([
            5,        # g_dot máximo
            5*RAD30,   # delta_dot máximo
            5        # vp_dot máximo
        ])

        # Terminal cost
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx
        self.ocp.constraints.lbx_e = self.ocp.constraints.lbx
        self.ocp.constraints.ubx_e = self.ocp.constraints.ubx 
        

        # Initial state constraint
        self.ocp.constraints.x0 = np.zeros(nx)
        #self.ocp.constraints.idxbx_0 = np.array([0,1,2,3,4,5]) # Constraint just vehicle state
        #self.ocp.constraints.lbx_0 = np.zeros(6)
        #self.ocp.constraints.ubx_0 = np.zeros(6)

        # --------------------------------- Non Linear constraints -----------------------------------#
        
        ## 1. Define the Constraint Expression (ONLY distance squared)
        #self.model.con_h_expr = ca.vertcat(e_d) 
        #self.model.con_h_expr_e = ca.vertcat(e_d)
#
        ## 2. Define the Bounds
        ## Keep the 3.0m leash (9.0 m^2)
        #max_dist_sq = 1.5**2
#
        #self.ocp.constraints.lh = np.array([0.0]) 
        #self.ocp.constraints.uh = np.array([max_dist_sq]) 
        #self.ocp.constraints.lh_e = self.ocp.constraints.lh
        #self.ocp.constraints.uh_e = self.ocp.constraints.uh
#
        ## 3. Configure Slacks (Now nh = 1)
        #nh = 1 
        #self.ocp.constraints.idxsh = np.arange(nh)
        #self.ocp.constraints.idxsh_e = np.arange(nh)
#
        ## Penalties: Using your DIST_PENALTY values
        #DIST_PENALTY_QUAD = 1e3
        #LINEAR_PENALTY = 1e2
#
        #self.ocp.cost.Zl = np.array([DIST_PENALTY_QUAD])
        #self.ocp.cost.Zu = np.array([DIST_PENALTY_QUAD])
        #self.ocp.cost.zl = np.array([LINEAR_PENALTY])
        #self.ocp.cost.zu = np.array([LINEAR_PENALTY])
#
        ## Apply to terminal node
        #self.ocp.cost.Zl_e = self.ocp.cost.Zl
        #self.ocp.cost.Zu_e = self.ocp.cost.Zu
        #self.ocp.cost.zl_e = self.ocp.cost.zl
        #self.ocp.cost.zu_e = self.ocp.cost.zu

        # --------------------------------- Slack constraints -----------------------------------#
        
        #
        ## Number of constraints
        #nbx = self.ocp.constraints.idxbx.shape[0] # Number of constrained states
        #nbu =  self.ocp.constraints.idxbu.shape[0] # Number of constrained controls
        #nsb = nbx + nbu + nh
        #
        ## --- For State (lbu/ubu bounds) ---
        #self.ocp.cost.Zl = SOFT_PENALTY * np.ones(nsb) 
        #self.ocp.cost.Zu = SOFT_PENALTY * np.ones(nsb) 
        #self.ocp.cost.Zl_0 = SOFT_PENALTY *np.ones(nbu)
        #self.ocp.cost.Zu_0 = SOFT_PENALTY *np.ones(nbu)
        #
        ## --- For Controls (lbu/ubu bounds) ---
        #self.ocp.cost.zl = SOFT_PENALTY * np.ones(nsb)
        #self.ocp.cost.zu = SOFT_PENALTY * np.ones(nsb)
        #self.ocp.cost.zl_0 = SOFT_PENALTY *np.ones(nbu)
        #self.ocp.cost.zu_0 = SOFT_PENALTY *np.ones(nbu)
        #
        ## --- Soft Constraint Indices ---
        #self.ocp.constraints.idxsbx = np.arange(nbx) # State Constraints
        #self.ocp.constraints.idxsbu = np.arange(nbu) # Control constraints
        #self.ocp.constraints.idxsh = np.arange(nh) # Non linear constraints
                    

    def _build_solver(self):
        # Inicialización de parámetros globales (Trayectoria)
        self.ocp.p_global_values = np.zeros(self.ocp.model.p_global.shape[0])
        self.ocp.code_export_directory = "/tmp/c_generated_code"
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json', verbose=False)

    def run(self, x0, x_window, u_window, param_vals):
        # Set the new initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Just set the whole window as is. 
        # acados internally handles the shift if you have warm_start enabled.
        for i in range(self.N):
            self.ocp_solver.set(i, "x", x_window[i])
            self.ocp_solver.set(i, "u", u_window[i])
        
        self.ocp_solver.set_p_global_and_precompute_dependencies(param_vals)
        return self.ocp_solver.solve(), \
            np.array([self.ocp_solver.get(i,"x") for i in range(self.N+1)]), \
            np.array([self.ocp_solver.get(i,"u") for i in range(self.N)])

