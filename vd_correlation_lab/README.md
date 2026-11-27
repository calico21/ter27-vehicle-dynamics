# Laboratorio de Correlación Multi-Vehículo y Multi-Modelo (`vd_correlation_lab`)

Motor modular de auditoría y correlación Sim-to-Real para los monoplazas de **Tecnun eRacing** sobre `src/mirena_sim/`.

## Arquitectura

```text
vd_correlation_lab/
├── run_lab.py                  # CLI: --vehicle [ter26_rwd|ter27_4wd|all] --model [full_14dof|fast_3dof|all]
├── dbc/TER.dbc                 # Base de datos CAN del equipo
├── core/
│   ├── log_parser.py           # Decodificación SocketCAN (.log + TER.dbc) y carga CSV
│   ├── signal_conditioning.py  # Calibración IMU, BPPS EMI, offset volante, TRQ_LIMIT(t), Butterworth 15 Hz
│   ├── godot_runner.py         # Ejecución paralela headless de src/mirena_sim
│   ├── metrics_engine.py       # r, R², RMSE, MAE, NRMSE, lag óptimo, coherencia γ²(f), Kus y fases
│   └── report_generator.py     # Generación de metrics.json e INFORME_CORRELACION.md
├── plotting/
│   ├── style.py                # Paleta Tecnun eRacing y exportación 300 DPI
│   ├── suite_01..08_*.py       # 8 suites por (vehículo, modelo)
│   ├── suite_09_cross_vehicle.py # Comparativa entre coches (TeR26 vs TeR27)
│   └── suite_10_cross_model.py   # Comparativa entre modelos (full_14dof vs fast_3dof)
├── results/
│   ├── cross_vehicle/          # Comparativas entre coches por modelo
│   └── cross_model/            # Comparativas 14-DOF vs 3-DOF por coche (gráficas + COMPARATIVA_MODELOS.md)
└── vehicles/
    ├── ter26_rwd/
    │   ├── sensor_calibration.json
    │   ├── logs/
    │   └── models/
    │       ├── full_14dof/ (results/ + plots/01..08)
    │       └── fast_3dof/  (results/ + plots/01..08)
    └── ter27_4wd/
        ├── sensor_calibration.json
        ├── logs/
        └── models/
            ├── full_14dof/ (results/ + plots/01..08)
            └── fast_3dof/  (results/ + plots/01..08)
```

## Uso rápido

```bash
# Evaluar TeR26 con ambos modelos y generar comparativa 14-DOF vs 3-DOF
python3 vd_correlation_lab/run_lab.py --vehicle ter26_rwd --model all

# Evaluar todos los vehículos y modelos (genera results/cross_model/ y results/cross_vehicle/)
python3 vd_correlation_lab/run_lab.py --vehicle all --model all
```
