# Reporte de Validacion Exhaustiva — `EcuTorqueVectoring4Wd` (TeR27 @ ~1.5 G)

- Tiempo total de simulacion en Godot: `127.8 s` (`dt = 5 ms`, `200 Hz`).
- Auditoria STM32F405 / FISTA: `P_dc <= 80 kW` valido en `100.00%` de los ticks (pico `80.00 kW`), par negativo `< 3 km/h`: `0`, residual medio Mz: `60.2 Nm`, tiempo paso TV: mean `0.437 ms`, p99 `0.738 ms`, max `1.798 ms`.

## Bloque 1: Dinamica Lateral, Bode y Neumáticos
| Test / Maniobra | Ctrl | Metricas Clave |
| :--- | :---: | :--- |
| DLC 15 m/s | `open` | max|beta|=2.99°, pass=True, max|ay|=15.59 m/s², t_stab=0.2 s |
| DLC 15 m/s | `v3` | max|beta|=0.43°, pass=True, max|ay|=12.5 m/s², t_stab=0.2 s |
| DLC 15 m/s | `tv4` | max|beta|=2.14°, pass=True, max|ay|=14.72 m/s², t_stab=0.2 s |
| DLC 18 m/s | `open` | max|beta|=5.02°, pass=True, max|ay|=16.76 m/s², t_stab=0.2 s |
| DLC 18 m/s | `v3` | max|beta|=0.84°, pass=True, max|ay|=14.15 m/s², t_stab=0.2 s |
| DLC 18 m/s | `tv4` | max|beta|=2.44°, pass=True, max|ay|=15.58 m/s², t_stab=0.2 s |
| DLC 20 m/s | `open` | max|beta|=10.14°, pass=True, max|ay|=17.85 m/s², t_stab=0.25 s |
| DLC 20 m/s | `v3` | max|beta|=1.23°, pass=True, max|ay|=15.29 m/s², t_stab=0.2 s |
| DLC 20 m/s | `tv4` | max|beta|=2.59°, pass=True, max|ay|=16.39 m/s², t_stab=0.2 s |
| Bode Chirp 0.2-3.5Hz | `open` | BW(-3dB)=3.25 Hz, Pico=2.61 dB, Lag={'0.5Hz': np.float64(0.0), '1.0Hz': np.float64(61.9), '2.0Hz': np.float64(61.7)} |
| Bode Chirp 0.2-3.5Hz | `v3` | BW(-3dB)=3.25 Hz, Pico=5.67 dB, Lag={'0.5Hz': np.float64(0.0), '1.0Hz': np.float64(12.3), '2.0Hz': np.float64(48.8)} |
| Bode Chirp 0.2-3.5Hz | `tv4` | BW(-3dB)=3.25 Hz, Pico=1.38 dB, Lag={'0.5Hz': np.float64(0.0), '1.0Hz': np.float64(28.1), '2.0Hz': np.float64(35.2)} |
| Horquilla 16° | `open` | vx_3s=27.71 m/s, R=37.07 m, std(eta)=0.1103 |
| Horquilla 16° | `v3` | vx_3s=0.85 m/s, R=0.33 m, std(eta)=0.0366 |
| Horquilla 16° | `tv4` | vx_3s=12.04 m/s, R=11.14 m, std(eta)=0.0695 |
| Trail-Braking | `open` | dist=17.26 m, max|ay|=14.86, max|wz|=2.419, max|beta|=82.98° |
| Trail-Braking | `v3` | dist=17.26 m, max|ay|=14.86, max|wz|=2.419, max|beta|=82.98° |
| Trail-Braking | `tv4` | dist=35.48 m, max|ay|=14.7, max|wz|=1.308, max|beta|=4.2° |

## Bloque 2: Cinematica Activa (56% Anti-Dive / 40% Anti-Squat)
- Turn-In `<40ms`: `dFz_front` activo `735.3 N` vs off `695.2 N`, pitch `0.343°` vs `0.357°`, r_dot `11.25` vs `10.67 rad/s²`.
- Squat Salida: `Delta pitch` TV4 `0.539°` vs Open `0.48°`, Fz tras fin `3555.6 N` vs `3477.0 N`.

## Bloque 3: Split-mu, mu-Jump y Asfalto Humedo
- Split-mu `open`: |y_end|=29.48 m, wz_max=0.398 rad/s, mu_hat L/R=1.5/1.5, EKF MAE=None m/s
- Split-mu `v3`: |y_end|=1.4 m, wz_max=0.134 rad/s, mu_hat L/R=1.5/1.5, EKF MAE=None m/s
- Split-mu `tv4`: |y_end|=0.02 m, wz_max=0.088 rad/s, mu_hat L/R=0.71/1.5, EKF MAE=0.044 m/s
- mu-Jump `open`: peak|beta|=11.59°, rec=0.32 s, mu_min=1.5
- mu-Jump `v3`: peak|beta|=1.15°, rec=0.0 s, mu_min=1.5
- mu-Jump `tv4`: peak|beta|=4.07°, rec=0.0 s, mu_min=0.73
- Wet 0.65 `open`: step max|ay|=11.31 m/s², step max|beta|=1.48°, hairpin vx_end=-6.75 m/s
- Wet 0.65 `v3`: step max|ay|=9.35 m/s², step max|beta|=0.93°, hairpin vx_end=-3.03 m/s
- Wet 0.65 `tv4`: step max|ay|=11.18 m/s², step max|beta|=4.9°, hairpin vx_end=9.51 m/s

## Bloque 4: Modos Formula Student
- Accel 75m `accel`: t75=3.805 s, 0-50=1.105 s, 0-100=2.73 s, max_k=0.248, Pmax=80.0 kW
- Accel 75m `auto`: t75=3.805 s, 0-50=1.1 s, 0-100=2.765 s, max_k=0.28, Pmax=80.0 kW
- Accel 75m `v3`: t75=3.855 s, 0-50=1.15 s, 0-100=2.775 s, max_k=1.427, Pmax=80.0 kW
- Accel 75m `open`: t75=3.905 s, 0-50=1.23 s, 0-100=2.85 s, max_k=3.428, Pmax=80.0 kW
- Skidpad `skidpad`: t_lap=5.836 s, vx_ss=9.83 m/s, ay_ss=10.6 m/s², trans_ov=0.07 rad/s
- Skidpad `auto`: t_lap=4.827 s, vx_ss=11.88 m/s, ay_ss=12.05 m/s², trans_ov=0.031 rad/s
- Skidpad `v3`: t_lap=4.942 s, vx_ss=11.6 m/s, ay_ss=12.35 m/s², trans_ov=0.105 rad/s
- Efficiency 60s `eff`: dist=1122.0 m, E_net=1168.4 kJ, E_regen=11.9 kJ, TF_loss=0.02 kJ, front_disc=33.4%
- Efficiency 60s `auto`: dist=1239.8 m, E_net=1564.0 kJ, E_regen=21.4 kJ, TF_loss=86.81 kJ, front_disc=7.5%
- Efficiency 60s `v3`: dist=616.3 m, E_net=1520.4 kJ, E_regen=14.3 kJ, TF_loss=0.0 kJ, front_disc=12.3%
- Tire Warmer 35 km/h: RMSE(vx)=0.098 m/s, P_therm=3.25 kW (Del=2.52 / Tras=0.73 kW), P_dc=8.08 kW

## Bloque 5 y 6: Guardias de Seguridad y Sensibilidad
- Curb-Strike: latencia recorte `0.0 ms`, T_FL aire `0.0 Nm`, pico recontacto `91.2 Nm`.
- Lift-Off: beta con atenuacion `3.96°` vs fija `2.9°`.
- Limp-Home: wheel fault EKF max err `0.17 m/s`, beta max `3.1°`; steer fault beta max `3.08°`.
- Sensibilidad `nom`: step ay=15.09 m/s², wz=1.027 rad/s, beta=5.3° | skid ay=11.99 m/s², vx=12.7 m/s
- Sensibilidad `mass_cg`: step ay=14.67 m/s², wz=1.014 rad/s, beta=5.24° | skid ay=11.76 m/s², vx=12.57 m/s
- Sensibilidad `mu_err`: step ay=15.06 m/s², wz=1.042 rad/s, beta=5.49° | skid ay=11.84 m/s², vx=12.6 m/s
- Sensibilidad `mu_imbal`: step ay=16.05 m/s², wz=0.879 rad/s, beta=4.45° | skid ay=12.08 m/s², vx=13.84 m/s