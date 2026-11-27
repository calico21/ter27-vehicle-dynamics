class_name LtsLivePlot
extends Control

## Live custom-drawn chart widget inside the LTS tab:
##   - Benchmark mode: Open-Diff vs TV4 Autocross vs TV4 Efficiency bars (Lap Time & E_22km)
##   - Skidpad / Autocross mode: Trial # vs Lap Time convergence with feasible/rejected markers & best line
##   - Efficiency mode: Pareto scatter (T_22km vs E_22km), flat 47.5 kW ref, cyan Pareto front, gold best J_FSG
##   - Telemetry Compare mode: Distance-domain Delta-t(s) + v(s) + Throttle/Brake overlay
##     comparing Driver/QSS vs CasADi validated reference, highlighting time-loss zones & clickable corners.

signal corner_clicked(corner_idx: int, corner_data: Dictionary)

enum PlotMode {
	BENCHMARK = 0,
	CONVERGENCE = 1,
	PARETO = 2,
	TELEMETRY_COMPARE = 3,
}

var plot_mode: int = PlotMode.BENCHMARK
var events: Array = []
var total_trials: int = 0
var best_lap_s: float = 0.0
var best_e_22km: float = 0.0
var best_j_fsg: float = 0.0
var best_params: Dictionary = {}
var feasible_count: int = 0
var rejected_count: int = 0
var is_done: bool = false
var summary_path: String = ""
var best_trace_csv: String = ""

# Distance-domain telemetry comparison state
var ref_waypoints: Array = []
var cmp_waypoints: Array = []
var corners: Array = []
var selected_corner_idx: int = -1
var ref_label: String = "CasADi"
var cmp_label: String = "QSS / Driver"
var _s_total_m: float = 100.0
var _delta_t_series: Array = [] # [{ "s": float, "dt": float, "v_ref_kmh": float, "v_cmp_kmh": float, "thr": float, "brk": float, "loss_rate": float }]
var _plot_rect: Rect2 = Rect2(44.0, 22.0, 300.0, 160.0)


func _init() -> void:
	custom_minimum_size = Vector2(360, 212)
	focus_mode = Control.FOCUS_NONE


func reset_plot(mode: int) -> void:
	plot_mode = mode
	events.clear()
	total_trials = 0
	best_lap_s = 0.0
	best_e_22km = 0.0
	best_j_fsg = 0.0
	best_params.clear()
	feasible_count = 0
	rejected_count = 0
	is_done = false
	summary_path = ""
	best_trace_csv = ""
	queue_redraw()


func set_selected_corner(idx: int) -> void:
	if idx >= -1 and idx < corners.size():
		selected_corner_idx = idx
		queue_redraw()


func set_telemetry_comparison(
	ref_pts: Array,
	cmp_pts: Array,
	corner_list: Array,
	p_ref_label: String = "CasADi",
	p_cmp_label: String = "QSS / Driver"
) -> void:
	ref_waypoints = ref_pts
	cmp_waypoints = cmp_pts
	corners = corner_list
	ref_label = p_ref_label
	cmp_label = p_cmp_label
	if selected_corner_idx >= corners.size():
		selected_corner_idx = 0 if not corners.is_empty() else -1
	_rebuild_telemetry_series()
	plot_mode = PlotMode.TELEMETRY_COMPARE
	queue_redraw()


func _rebuild_telemetry_series() -> void:
	_delta_t_series.clear()
	if ref_waypoints.is_empty():
		_s_total_m = 100.0
		return

	# Build cumulative time for ref
	var ref_s: Array[float] = []
	var ref_t: Array[float] = []
	var ref_v: Array[float] = []
	var ref_thr: Array[float] = []
	var ref_brk: Array[float] = []
	var t_acc := 0.0
	var prev_s := 0.0
	for i in range(ref_waypoints.size()):
		var p: Dictionary = ref_waypoints[i]
		var s_i := float(p.get("s", p.get("s_m", float(i) * 2.5)))
		var v_i := maxf(float(p.get("v_ref", p.get("v_mps", 10.0))), 1.0)
		if i > 0:
			t_acc += maxf(s_i - prev_s, 0.0) / v_i
		prev_s = s_i
		ref_s.append(s_i)
		ref_t.append(t_acc)
		ref_v.append(v_i)
		ref_thr.append(clampf(float(p.get("throttle", 0.0)), 0.0, 1.0))
		ref_brk.append(clampf(float(p.get("brake", 0.0)), 0.0, 1.0))

	_s_total_m = maxf(ref_s[ref_s.size() - 1], 10.0)

	# Build cumulative time for cmp (either explicit "t" or integrated from v)
	var cmp_s: Array[float] = []
	var cmp_t: Array[float] = []
	var cmp_v: Array[float] = []
	var has_cmp := not cmp_waypoints.is_empty()
	if has_cmp:
		var tc_acc := 0.0
		var prev_sc := 0.0
		var has_t_field := (cmp_waypoints[0] is Dictionary) and (cmp_waypoints[0] as Dictionary).has("t")
		for i in range(cmp_waypoints.size()):
			var p: Dictionary = cmp_waypoints[i]
			var s_i := float(p.get("s", p.get("s_m", float(i) * 2.5)))
			var v_i := maxf(float(p.get("v_ref", p.get("v_mps", 10.0))), 1.0)
			if has_t_field:
				tc_acc = float(p.get("t", tc_acc))
				if i > 0 and not p.has("v_mps") and not p.has("v_ref"):
					var ds_c := maxf(s_i - prev_sc, 0.0)
					var dt_c := maxf(tc_acc - cmp_t[i - 1], 1e-3)
					v_i = clampf(ds_c / dt_c, 1.0, 45.0)
			else:
				if i > 0:
					tc_acc += maxf(s_i - prev_sc, 0.0) / v_i
			prev_sc = s_i
			cmp_s.append(s_i)
			cmp_t.append(tc_acc)
			cmp_v.append(v_i)

	var prev_dt := 0.0
	for i in range(ref_s.size()):
		var s_i := ref_s[i]
		var dt_val := 0.0
		var v_cmp_i := ref_v[i]
		if has_cmp and cmp_s.size() >= 2:
			var t_c := _interp_1d(cmp_s, cmp_t, s_i)
			v_cmp_i = _interp_1d(cmp_s, cmp_v, s_i)
			dt_val = t_c - ref_t[i]
		var loss_rate := (dt_val - prev_dt) if i > 0 else 0.0
		prev_dt = dt_val
		_delta_t_series.append({
			"s": s_i,
			"dt": dt_val,
			"v_ref_kmh": ref_v[i] * 3.6,
			"v_cmp_kmh": v_cmp_i * 3.6,
			"thr": ref_thr[i],
			"brk": ref_brk[i],
			"loss_rate": loss_rate,
		})


static func _interp_1d(xs: Array[float], ys: Array[float], x_q: float) -> float:
	var n := xs.size()
	if n == 0:
		return 0.0
	if x_q <= xs[0]:
		return ys[0]
	if x_q >= xs[n - 1]:
		return ys[n - 1]
	var lo := 0
	var hi := n - 1
	while hi - lo > 1:
		var mid := (lo + hi) >> 1
		if xs[mid] <= x_q:
			lo = mid
		else:
			hi = mid
	var u := clampf((x_q - xs[lo]) / maxf(xs[hi] - xs[lo], 1e-5), 0.0, 1.0)
	return lerpf(ys[lo], ys[hi], u)


func _gui_input(event: InputEvent) -> void:
	if plot_mode != PlotMode.TELEMETRY_COMPARE or corners.is_empty():
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
			var pos := mb.position
			if _plot_rect.has_point(pos):
				var frac := clampf((pos.x - _plot_rect.position.x) / maxf(_plot_rect.size.x, 1.0), 0.0, 1.0)
				var s_click := frac * _s_total_m
				var best_idx := -1
				var best_d := 1e9
				for ci in range(corners.size()):
					var c: Dictionary = corners[ci]
					var s_apx := float(c.get("s_apex", c.get("s_brake", 0.0)))
					var s_st := float(c.get("s_start", s_apx - 15.0))
					var s_ex := float(c.get("s_exit", s_apx + 15.0))
					var d := 0.0 if (s_click >= s_st and s_click <= s_ex) else absf(s_click - s_apx)
					if d < best_d:
						best_d = d
						best_idx = ci
				if best_idx >= 0:
					selected_corner_idx = best_idx
					queue_redraw()
					corner_clicked.emit(best_idx, corners[best_idx])


func ingest_jsonl_line(line: String) -> Dictionary:
	var txt := line.strip_edges()
	if txt.is_empty():
		return {}
	var parsed: Variant = JSON.parse_string(txt)
	if typeof(parsed) != TYPE_DICTIONARY:
		return {}
	var ev: Dictionary = parsed
	if bool(ev.get("done", false)):
		is_done = true
		summary_path = String(ev.get("summary_path", ""))
		best_trace_csv = String(ev.get("best_trace_csv", ""))
		queue_redraw()
		return ev

	events.append(ev)
	total_trials = maxi(total_trials, int(ev.get("total_trials", 0)))
	var feas := bool(ev.get("feasible", false))
	if feas:
		feasible_count += 1
	else:
		rejected_count += 1

	var lap_v := float(ev.get("lap_s", 0.0))
	var e_v := float(ev.get("e_22km_kwh", 0.0))
	var j_v := float(ev.get("j_fsg", 0.0))
	var stg := String(ev.get("stage", ""))
	if stg == "efficiency":
		plot_mode = PlotMode.PARETO
		if bool(ev.get("is_best", false)) or (feas and (best_j_fsg <= 0.0 or j_v < best_j_fsg)):
			best_j_fsg = j_v
			best_lap_s = lap_v
			best_e_22km = e_v
			best_params = ev.get("params", {}) if ev.get("params") is Dictionary else {}
	elif stg == "benchmark":
		plot_mode = PlotMode.BENCHMARK
		if String(ev.get("key", "")) == "tv4_auto" or best_lap_s <= 0.0:
			best_lap_s = lap_v
			best_e_22km = e_v
	else:
		plot_mode = PlotMode.CONVERGENCE
		if bool(ev.get("is_best", false)) or (feas and (best_lap_s <= 0.0 or lap_v < best_lap_s)):
			best_lap_s = lap_v
			best_params = ev.get("params", {}) if ev.get("params") is Dictionary else {}

	queue_redraw()
	return ev


func _draw() -> void:
	var w := size.x
	var h := size.y
	if w < 40.0 or h < 40.0:
		return

	draw_rect(Rect2(0, 0, w, h), Color(0.08, 0.10, 0.14, 0.95), true)
	draw_rect(Rect2(0, 0, w, h), Color(0.24, 0.30, 0.38, 1.0), false, 1.0)

	var pad_l := 44.0
	var pad_r := 12.0
	var pad_t := 20.0
	var pad_b := 20.0
	var pw := maxf(w - pad_l - pad_r, 10.0)
	var ph := maxf(h - pad_t - pad_b, 10.0)
	_plot_rect = Rect2(pad_l, pad_t, pw, ph)
	var font := ThemeDB.fallback_font

	if plot_mode == PlotMode.TELEMETRY_COMPARE:
		_draw_telemetry_compare(pad_l, pad_t, pw, ph, font)
		return

	if events.is_empty():
		if font != null:
			draw_string(font, Vector2(pad_l + 8.0, pad_t + 0.5 * ph), "Ready. Select solver and click Run.", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.72, 0.78, 0.86))
		return

	# Axes
	var ax_col := Color(0.45, 0.52, 0.62, 0.8)
	draw_line(Vector2(pad_l, pad_t), Vector2(pad_l, pad_t + ph), ax_col, 1.0)
	draw_line(Vector2(pad_l, pad_t + ph), Vector2(pad_l + pw, pad_t + ph), ax_col, 1.0)

	if plot_mode == PlotMode.BENCHMARK:
		_draw_benchmark(pad_l, pad_t, pw, ph, font)
	elif plot_mode == PlotMode.CONVERGENCE:
		_draw_convergence(pad_l, pad_t, pw, ph, font)
	else:
		_draw_pareto(pad_l, pad_t, pw, ph, font)


func _draw_telemetry_compare(x0: float, y0: float, pw: float, ph: float, font: Font) -> void:
	if _delta_t_series.size() < 2:
		if font != null:
			draw_string(font, Vector2(x0 + 6.0, y0 + 0.5 * ph), "No telemetry loaded. Run CasADi or select a lap.", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.72, 0.78, 0.86))
		return

	if font != null:
		draw_string(font, Vector2(x0, y0 - 5.0),
			"Delta-t(s) & v(s): %s (cyan) vs %s (amber) [click corner]" % [ref_label, cmp_label],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.84, 0.90, 0.98))

	# Band heights: Delta-t (32%), v(s) (46%), Throttle/Brake (22%)
	var h_dt := ph * 0.31
	var h_v := ph * 0.45
	var h_tb := ph - h_dt - h_v - 8.0
	var y_dt := y0
	var y_v := y_dt + h_dt + 4.0
	var y_tb := y_v + h_v + 4.0

	# Corner highlight columns
	for ci in range(corners.size()):
		var c: Dictionary = corners[ci]
		var s_st := float(c.get("s_start", float(c.get("s_brake", 0.0))))
		var s_ex := float(c.get("s_exit", s_st + 20.0))
		var cx0 := x0 + clampf(s_st / _s_total_m, 0.0, 1.0) * pw
		var cx1 := x0 + clampf(s_ex / _s_total_m, 0.0, 1.0) * pw
		var is_sel := (ci == selected_corner_idx)
		var bg_c := Color(0.20, 0.76, 0.96, 0.22) if is_sel else Color(0.30, 0.38, 0.48, 0.12)
		draw_rect(Rect2(cx0, y0, maxf(cx1 - cx0, 3.0), ph), bg_c, true)
		if is_sel:
			draw_rect(Rect2(cx0, y0, maxf(cx1 - cx0, 3.0), ph), Color(0.25, 0.90, 1.0, 0.85), false, 1.2)
		if font != null and (is_sel or (ci % maxi(1, int(ceil(float(corners.size()) / 8.0))) == 0)):
			var cname := "C%d" % (ci + 1)
			draw_string(font, Vector2(cx0 + 1.0, y0 + ph + 12.0), cname, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.96, 0.86, 0.30) if is_sel else Color(0.68, 0.76, 0.84))

	# Band borders
	var sep_col := Color(0.32, 0.38, 0.48, 0.7)
	draw_line(Vector2(x0, y_v - 2.0), Vector2(x0 + pw, y_v - 2.0), sep_col, 1.0)
	draw_line(Vector2(x0, y_tb - 2.0), Vector2(x0 + pw, y_tb - 2.0), sep_col, 1.0)

	# 1. Delta-t(s) band
	var dt_abs_max := 0.25
	var v_max_kmh := 80.0
	for pt_v in _delta_t_series:
		var pt: Dictionary = pt_v
		dt_abs_max = maxf(dt_abs_max, absf(float(pt["dt"])))
		v_max_kmh = maxf(v_max_kmh, maxf(float(pt["v_ref_kmh"]), float(pt["v_cmp_kmh"])))
	dt_abs_max *= 1.15
	v_max_kmh = ceil(v_max_kmh / 10.0) * 10.0

	var y_dt_zero := y_dt + h_dt * 0.5
	draw_line(Vector2(x0, y_dt_zero), Vector2(x0 + pw, y_dt_zero), Color(0.50, 0.58, 0.68, 0.65), 1.0)

	var dt_pts := PackedVector2Array()
	var v_ref_pts := PackedVector2Array()
	var v_cmp_pts := PackedVector2Array()
	var thr_pts := PackedVector2Array()
	var brk_pts := PackedVector2Array()

	for i in range(_delta_t_series.size()):
		var pt: Dictionary = _delta_t_series[i]
		var px := x0 + clampf(float(pt["s"]) / _s_total_m, 0.0, 1.0) * pw
		var dt_v := float(pt["dt"])
		var py_dt := y_dt_zero - clampf(dt_v / dt_abs_max, -1.0, 1.0) * (h_dt * 0.45)
		dt_pts.append(Vector2(px, py_dt))

		# Time-loss zone shading (where cmp loses time vs ref)
		if i > 0:
			var prev_x := dt_pts[i - 1].x
			var lr := float(pt["loss_rate"])
			if lr > 0.002:
				draw_rect(Rect2(prev_x, y_dt, maxf(px - prev_x, 1.0), h_dt), Color(0.95, 0.20, 0.24, 0.18), true)
			elif lr < -0.002:
				draw_rect(Rect2(prev_x, y_dt, maxf(px - prev_x, 1.0), h_dt), Color(0.14, 0.86, 0.46, 0.14), true)

		var py_vr := y_v + h_v - clampf(float(pt["v_ref_kmh"]) / v_max_kmh, 0.0, 1.0) * h_v
		var py_vc := y_v + h_v - clampf(float(pt["v_cmp_kmh"]) / v_max_kmh, 0.0, 1.0) * h_v
		v_ref_pts.append(Vector2(px, py_vr))
		v_cmp_pts.append(Vector2(px, py_vc))

		var py_th := y_tb + h_tb - float(pt["thr"]) * h_tb
		var py_br := y_tb + h_tb - float(pt["brk"]) * h_tb
		thr_pts.append(Vector2(px, py_th))
		brk_pts.append(Vector2(px, py_br))

	draw_polyline(dt_pts, Color(0.96, 0.84, 0.24, 0.95), 1.5)
	if not cmp_waypoints.is_empty():
		draw_polyline(v_cmp_pts, Color(0.98, 0.66, 0.18, 0.88), 1.4)
	draw_polyline(v_ref_pts, Color(0.18, 0.88, 0.98, 0.96), 1.7)
	draw_polyline(thr_pts, Color(0.14, 0.88, 0.46, 0.90), 1.3)
	draw_polyline(brk_pts, Color(0.98, 0.22, 0.26, 0.92), 1.3)

	if font != null:
		draw_string(font, Vector2(3.0, y_dt + 10.0), "+%.1fs" % dt_abs_max, HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.75, 0.80, 0.88))
		draw_string(font, Vector2(3.0, y_dt + h_dt - 1.0), "-%.1fs" % dt_abs_max, HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.75, 0.80, 0.88))
		draw_string(font, Vector2(3.0, y_v + 10.0), "%.0f" % v_max_kmh, HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.75, 0.80, 0.88))
		draw_string(font, Vector2(3.0, y_v + h_v - 1.0), "km/h", HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.65, 0.72, 0.80))
		draw_string(font, Vector2(3.0, y_tb + h_tb - 2.0), "T/B", HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.65, 0.72, 0.80))


func _draw_benchmark(x0: float, y0: float, pw: float, ph: float, font: Font) -> void:
	if font != null:
		draw_string(font, Vector2(x0, y0 - 6.0), "FSG 2023 Live Benchmark: Lap Time [s] / E_22km [kWh]", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.84, 0.90, 0.98))
	var max_t := 85.0
	var n := mini(events.size(), 3)
	var slot_w := pw / 3.0
	var bar_w := minf(slot_w * 0.48, 42.0)
	var cols := [Color(0.62, 0.68, 0.76), Color(0.16, 0.82, 0.48), Color(0.24, 0.76, 0.96)]
	for i in range(n):
		var ev: Dictionary = events[i]
		var lap_s := float(ev.get("lap_s", 0.0))
		var e_kwh := float(ev.get("e_22km_kwh", 0.0))
		var bh := clampf(lap_s / max_t, 0.05, 1.0) * (ph - 18.0)
		var bx := x0 + (float(i) + 0.5) * slot_w - 0.5 * bar_w
		var by := y0 + ph - bh
		draw_rect(Rect2(bx, by, bar_w, bh), cols[i % cols.size()], true)
		if font != null:
			draw_string(font, Vector2(bx - 6.0, by - 4.0), "%.2fs" % lap_s, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.95, 0.96, 0.98))
			draw_string(font, Vector2(bx - 6.0, y0 + ph + 13.0), "%.2fkWh" % e_kwh, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.75, 0.84, 0.92))


func _draw_convergence(x0: float, y0: float, pw: float, ph: float, font: Font) -> void:
	if font != null:
		draw_string(font, Vector2(x0, y0 - 6.0), "Convergence: Trial vs Lap Time [s] (Best=%.3fs)" % best_lap_s, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.84, 0.90, 0.98))
	var y_min := INF
	var y_max := -INF
	for ev_v in events:
		var v := float((ev_v as Dictionary).get("lap_s", 0.0))
		if is_finite(v) and v > 0.0:
			y_min = minf(y_min, v)
			y_max = maxf(y_max, v)
	if not is_finite(y_min) or y_max <= y_min:
		y_min = 0.0; y_max = 80.0
	var pad_y := maxf((y_max - y_min) * 0.15, 0.05)
	y_min -= pad_y
	y_max += pad_y

	var denom_x := maxf(float(maxi(total_trials - 1, events.size() - 1)), 1.0)
	var cur_best := INF
	var prev_pt := Vector2.ZERO
	var has_prev := false
	for i in range(events.size()):
		var ev: Dictionary = events[i]
		var lap_v := float(ev.get("lap_s", y_min))
		var feas := bool(ev.get("feasible", false))
		var px := x0 + (float(i) / denom_x) * pw
		var py := y0 + ph - clampf((lap_v - y_min) / maxf(y_max - y_min, 1e-4), 0.0, 1.0) * ph
		var c := Color(0.18, 0.86, 0.48) if feas else Color(0.92, 0.30, 0.28)
		draw_circle(Vector2(px, py), 4.0, c)
		if feas and lap_v < cur_best:
			cur_best = lap_v
		if is_finite(cur_best):
			var by := y0 + ph - clampf((cur_best - y_min) / maxf(y_max - y_min, 1e-4), 0.0, 1.0) * ph
			var bp := Vector2(px, by)
			if has_prev:
				draw_line(prev_pt, Vector2(px, prev_pt.y), Color(0.98, 0.78, 0.18), 1.6)
				draw_line(Vector2(px, prev_pt.y), bp, Color(0.98, 0.78, 0.18), 1.6)
			prev_pt = bp
			has_prev = true
	if font != null:
		draw_string(font, Vector2(4.0, y0 + 10.0), "%.2f" % y_max, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.70, 0.76, 0.84))
		draw_string(font, Vector2(4.0, y0 + ph), "%.2f" % y_min, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.70, 0.76, 0.84))


func _draw_pareto(x0: float, y0: float, pw: float, ph: float, font: Font) -> void:
	if font != null:
		draw_string(font, Vector2(x0, y0 - 6.0), "Pareto: T_22km [s] vs E_22km [kWh] (Min J=%.2fe6)" % (best_j_fsg * 1e-6), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.84, 0.90, 0.98))
	var t_min := INF; var t_max := -INF
	var e_min := INF; var e_max := -INF
	for ev_v in events:
		var ev: Dictionary = ev_v
		var tv := float(ev.get("t_22km_s", 0.0))
		var ev_e := float(ev.get("e_22km_kwh", 0.0))
		if tv > 0.0 and ev_e > 0.0:
			t_min = minf(t_min, tv); t_max = maxf(t_max, tv)
			e_min = minf(e_min, ev_e); e_max = maxf(e_max, ev_e)
	if not is_finite(t_min) or t_max <= t_min:
		t_min = 1350.0; t_max = 1480.0
	if not is_finite(e_min) or e_max <= e_min:
		e_min = 4.0; e_max = 5.4
	t_min -= 8.0; t_max += 8.0
	e_min -= 0.12; e_max += 0.12

	# 5.20 kWh FSG constraint line if inside view
	if 5.20 >= e_min and 5.20 <= e_max:
		var y_lim := y0 + ph - ((5.20 - e_min) / (e_max - e_min)) * ph
		draw_line(Vector2(x0, y_lim), Vector2(x0 + pw, y_lim), Color(0.92, 0.28, 0.28, 0.65), 1.0)

	# Build non-dominated Pareto front among feasible candidates
	var feas_pts: Array = []
	for ev_v in events:
		var ev: Dictionary = ev_v
		var tv := float(ev.get("t_22km_s", 0.0))
		var ev_e := float(ev.get("e_22km_kwh", 0.0))
		var px := x0 + clampf((tv - t_min) / maxf(t_max - t_min, 1e-4), 0.0, 1.0) * pw
		var py := y0 + ph - clampf((ev_e - e_min) / maxf(e_max - e_min, 1e-4), 0.0, 1.0) * ph
		if bool(ev.get("is_flat_ref", false)):
			draw_rect(Rect2(px - 4.0, py - 4.0, 8.0, 8.0), Color(0.92, 0.34, 0.24), true)
		else:
			var col := Color(0.22, 0.84, 0.96) if bool(ev.get("feasible", false)) else Color(0.55, 0.60, 0.68, 0.7)
			draw_circle(Vector2(px, py), 3.6, col)
		if bool(ev.get("feasible", false)) and tv > 0.0 and ev_e > 0.0:
			feas_pts.append({"t": tv, "e": ev_e, "p": Vector2(px, py), "j": float(ev.get("j_fsg", 1e12))})

	if feas_pts.size() >= 2:
		feas_pts.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return float(a["t"]) < float(b["t"]))
		var front: Array = []
		var min_e := INF
		for fp in feas_pts:
			if float(fp["e"]) < min_e:
				min_e = float(fp["e"])
				front.append(fp["p"])
		for k in range(1, front.size()):
			draw_line(front[k - 1], front[k], Color(0.22, 0.88, 0.98, 0.90), 1.6)

	if best_j_fsg > 0.0 and best_e_22km > 0.0:
		for fp in feas_pts:
			if absf(float(fp["j"]) - best_j_fsg) < 1.0:
				var bp: Vector2 = fp["p"]
				draw_circle(bp, 5.5, Color(1.0, 0.84, 0.16))
				break

	if font != null:
		draw_string(font, Vector2(4.0, y0 + 10.0), "%.2f" % e_max, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.70, 0.76, 0.84))
		draw_string(font, Vector2(4.0, y0 + ph), "%.2f" % e_min, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.70, 0.76, 0.84))
