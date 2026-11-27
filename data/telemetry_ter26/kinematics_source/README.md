# Kinematics Source Data

This directory documents the CAD and spreadsheet kinematics source data for Tecnun eRacing vehicles.

- **TeR26 / TeR21 Reference Hardpoints**: The 3D double-wishbone suspension hardpoints from `TER21 FINAL.xlsx` (front and rear pushrod/pullrod bellcrank geometry, steering rack, upright ball joints, U-bar ARB) are codified directly in:
  - `vd_correlation_lab/kinematics/ter26_double_wishbone_3d.py`
  - `vd_correlation_lab/kinematics/export_kinematics_casadi_lut.py`
  - `vd_correlation_lab/kinematics/export_ter26_godot_bundle.py`

Place any future raw CAD / spreadsheet hardpoint exports (`.xlsx`, `.csv`, `.json`) in this directory.
