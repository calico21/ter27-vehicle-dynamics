# Solver `full_14dof` (`Full14DofModel.gd`)

Modelo vehicular de alta fidelidad de **14 grados de libertad**:
- **Chasis (6 GDL):** posición $(X, Y, Z)$, orientación Euler Z-Y-X $(\phi, \theta, \psi)$ y velocidades $(v_x, v_y, v_z, w_x, w_y, w_z)$.
- **Suspensión (4 GDL):** desplazamientos y velocidades verticales por esquina $(\Delta z_i, \Delta \dot{z}_i)$ con muelles, amortiguadores asimétricos, *bumpstops*, ARBs, *anti-dive/anti-squat* y cinemática de camber/toe.
- **Ruedas (4 GDL):** velocidades de giro $\omega_i$ con relajación transitoria de carcasa y Pacejka MF 6.2.

## Paso e integración
- **Integrador:** RK2 (Heun) en `step_standalone()`, o acoplado al `RigidBody3D` mediante `step_forces()`.
- **Frecuencia recomendada:** `1000 Hz` (`dt = 0.001 s`) en validación offline; `400 Hz` en escena Godot 3D.
- **Uso:** Dinámica Vehicular, ajuste de *set-up* mecánico, análisis de amortiguadores y validación Sim-to-Real de referencia.
