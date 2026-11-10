"""
Tire Model Validation Framework — Core Package
"""
from .metrics import compute_metrics, decompose_error, bootstrap_confidence_interval
from .features import extract_lateral_features, extract_aligning_features, extract_camber_features, extract_vertical_features
from .evaluator import TireModelEvaluator
from .style import apply_theme, add_metrics_box, add_characterization_badge, COLOR_EXP, COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB, STYLE_M1, STYLE_M2, STYLE_M3, LOAD_LEVELS_INFO, LOAD_COLORS
