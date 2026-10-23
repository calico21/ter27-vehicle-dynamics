# IMU comparison: real bag vs simulator

- real IMU: `real_imu.csv` (40344 rows, 99.95 Hz, frame `car/cog`)
- sim IMU: `sim_imu.csv` (47001 rows, 100.00 Hz, frame `car/cog`)
- real control: 151015 rows, onset at t_rel=123.345s
- sim control: 151015 rows, onset at t_rel=128.201s
- compared window: tau = 0..281.0s, 28103 samples at 100 Hz

## Static bias (pre-onset mean)

| axis | real | sim | difference |
|---|---:|---:|---:|
| ax | +0.0897 | -0.0002 | -0.0899 |
| ay | -0.0056 | +0.0002 | +0.0058 |
| az | -0.7315 | -0.0003 | +0.7312 |
| gx | -0.0001 | -0.0000 | +0.0000 |
| gy | -0.0000 | -0.0000 | +0.0000 |
| gz | +0.0001 | +0.0000 | -0.0001 |

## Dynamic metrics (main drive, tau >= 20 s)

| signal | real std | sim std | RMSE | corr | gain sim/real | lag [s] |
|---|---:|---:|---:|---:|---:|---:|
| ax | 2.1230 | 5.4983 | 5.6338 | 0.128 | 0.333 | -0.150 |
| ay | 7.0029 | 5.8486 | 8.8290 | 0.065 | 0.054 | -0.200 |
| az | 3.2635 | 4.0039 | 5.1515 | 0.006 | 0.007 | -43.130 |
| gx | 0.2250 | 0.2652 | 0.3457 | 0.012 | 0.014 | -49.360 |
| gy | 0.0607 | 0.2115 | 0.2205 | -0.007 | -0.024 | +3.740 |
| gz | 0.5271 | 0.2029 | 0.4750 | 0.436 | 0.168 | -6.470 |

## Integrated quantities, tau >= 0

- yaw change: real -0.236 rad, sim -18.106 rad (ratio 76.591)
- longitudinal Δv: real -43.494 m/s, sim +6.377 m/s (ratio -0.147)

Plots: `plots/accel_vs_time.png`, `plots/gyro_vs_time.png`, `plots/accel_psd.png`, `plots/gyro_psd.png`, `plots/yaw_integral.png`
