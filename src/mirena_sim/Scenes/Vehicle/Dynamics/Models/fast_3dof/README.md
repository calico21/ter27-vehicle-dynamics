# Solver `fast_3dof` (`Fast3DofModel.gd`)

Modelo vehicular simplificado de **7 grados de libertad efectivos (chasis plano 3-DOF + 4 ruedas)** diseñado para **Driverless** y HIL/NMPC en tiempo real:
- **Estados ($\mathbf{x} \in \mathbb{R}^{10}$):** posición y guiñada plana $(X, Y, \psi)$, velocidades en ejes cuerpo $(v_x, v_y, w_z)$ y giro de las 4 ruedas $(\omega_{FL}, \omega_{FR}, \omega_{RL}, \omega_{RR})$.
- **Transferencia de carga cuasi-estática:** aceleraciones filtradas $\tilde{a}_x, \tilde{a}_y$ ($\tau_{\text{wt}} = 35\text{ ms}$), *downforce* repartido por `aero_balance_front`, transferencia longitudinal $\Delta F_{z,\text{long}}$ y lateral por eje mediante el **LLTD** real calculado desde `suspension.json` (muelles, *Motion Ratios*, ARBs y *roll centers*), con `max(Fz_i, 0.0)`.
- **Subsistemas compartidos:** Pacejka MF 6.2, `DrivetrainModel`, `EcuTorqueVectoringV3`, `BrakeModel` y `AeroModel` desde la misma fuente `Config/vehicles/<id>/`.

## Paso e integración
- **Integrador:** RK2 (Heun) con sub-stepping de giro de rueda para estabilidad a baja velocidad y anclaje limpio en parado.
- **Frecuencia recomendada:** `200 Hz` (`dt = 0.005 s`).
- **Uso:** Simulación Driverless desatendida (`--fidelity=fast_3dof`), pruebas de planificación/control y barridos rápidos.
