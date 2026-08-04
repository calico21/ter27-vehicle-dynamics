# mirena_perception

Colourless cone detection from the LiDAR alone. The cloud is levelled, Patchwork++
takes the ground out, what is left is cut down to the band of heights a cone can
occupy, PCL's Euclidean cluster extraction clusters it, and a size-and-placement
filter keeps the clusters shaped like Formula Student cones. Every height is
measured against a ground map built from that same frame, not against a plane.

Written as a drop-in replacement for `mirena_perception`: same input cloud, same
`/perception/cones`, same heartbeat. Swap them by changing which package the
mission launch file names.

```python
Node(
    package='mirena_perception',
    executable='mirena_perception',
    name='mirena_perception',
    parameters=[{'use_sim_time': use_sim_time}],
),
```

## Interface

| Direction | Topic | Type |
| --- | --- | --- |
| in | `/sensors/lidar/points` | `sensor_msgs/PointCloud2` |
| out | `/perception/cones` | `mirena_common/EntityList` in `car/cog` |
| out | `health` | `mirena_common/Heartbeat` at 10 Hz |
| debug | `/perception/debug/clustered_points` | accepted cone points, coloured per cone |
| debug | `/perception/debug/nonground_points` | all non-ground points; grey means rejected |

Both debug topics are published only when something is subscribed, and only when
`enable_debug` is true.

### Differences from `mirena_perception`

- **No colour.** Every cone comes out as `unknown_cone` with confidence `1.0`. A
  cluster either passes the geometric filter or is dropped, so there is no graded
  score to report, and SLAM's colour vote has nothing to accumulate. Planning
  that relies on left/right colour will not work off this node alone.
- **No camera.** Neither `/sensors/camera/*` nor a GPU is touched, so there is no
  time synchroniser and nothing to wait for at startup.
- **The sensor's field of view, not the camera's.** Whatever the LiDAR sees, out
  to `ground_max_range`, is a candidate. On the RoboSense M1 that is the forward
  120° × 25°, so this is not a 360° detector; on a spinning sensor it would be.
  Either way there is a near blind spot: a sensor at height `h` tilted down by
  `t` degrees first sees the track surface at about `h / tan(t + half the
  vertical FOV)`, roughly 2.5 m for the M1 as mounted.
- Debug clouds are in `car/cog`, not the LiDAR frame, because that is where
  everything downstream of ground removal happens.

## Parameters

The two that have to match the real car:

| Parameter | Default | |
| --- | --- | --- |
| `sensor_height` | `1.0` | LiDAR height above the track, in metres. Patchwork++ measures every elevation against it. Getting it wrong is the first thing to check when nothing is detected. |
| `ground_max_range` | `50.0` | Detection radius. Patchwork++ *discards* points outside `[ground_min_range, ground_max_range]` rather than calling them non-ground, so this band is the node's field of view. |

Ground removal: `ground_min_range` (`0.5`), `remove_reflected_noise` (`true`,
needs an intensity channel and is skipped without one), `rnr_intensity_thr`
(`0.2`). That last one is in whatever units the driver publishes intensity in:
drivers disagree. The RoboSense driver here normalises to `[0, 1]`, which is what
Patchwork++'s default assumes; others report `0-255`. Check yours before changing
it — set too high, it stops being a noise filter and simply deletes every
steeply-downward point, which is the whole near field.

Patchwork++'s own knobs, all at its defaults except the sector counts:
`ground_num_min_pts` (`10`), `ground_thickness` (`0.125`), `ground_seed_margin`
(`0.125`), `ground_num_lpr` (`20`), `ground_num_iter` (`3`),
`ground_uprightness_thr` (`0.707`), `ground_vertical_plane_fit` (`true`),
`ground_temporal_revert` (`true`), `ground_sectors_each_zone`
(`[48, 96, 162, 96]`) and `ground_rings_each_zone` (`[2, 4, 4, 4]`).

`ground_num_min_pts` is the one to reach for when ground survives into the
detections: a patch holding fewer points than this is not fitted at all, and
Patchwork++ hands the whole patch on as non-ground. `ground_thickness` is how far
a point may sit off its patch plane and still be ground. The two zone vectors
must have exactly four entries, innermost zone first, because Patchwork++ indexes
all four regardless of what its `num_zones` says; a wrong length is logged and
ignored. Sectors divide the full circle, so a narrow field of view does not make
them coarser, it just leaves most of them empty — the counts here are finer than
upstream's `[16, 32, 54, 32]` because those were drawn for a 64-beam spinning
sensor and a 0.2° one puts enough points in a third of the area.

Track surface: `ground_cell_size` (`1.0`). Heights are measured against a map
built each frame from the points Patchwork++ called ground, cell by cell, rather
than against one flat plane. Where a cell has no ground return at all — the
shadow behind a barrier, a stretch the vertical field of view skips — it
extrapolates along the nearest patch plane Patchwork++ fitted, from
`getCenters()` and `getNormals()`, so the local slope is carried rather than
assumed flat. See [Mounting](#mounting) for why that matters.

Elevation band and ego box, applied per point before clustering: `min_elevation`
(`0.05`), `max_elevation` (`1.00`), `ego_half_length` (`1.8`), `ego_half_width`
(`1.0`). The floor is what removes below-floor multipath and the ground patches
ground removal leaves behind; both would otherwise cluster into perfectly
cone-shaped detections. The ceiling has to stay above `max_height`, and that is
what keeps it safe: anything reaching it is by definition taller than a cone, so
a barrier is still rejected as too tall rather than cut down into something
cone-sized. The ego box is the car looking at itself, which on a hoop-mounted
sensor is a large permanent slab at cone height a few tens of centimetres away —
half the surviving cloud on the test bag.

Clustering: `voxel_leaf` (`0.05`), `cluster_tolerance` (`0.30`). Two points join
the same cluster when they sit within `cluster_tolerance` of each other, so it
has to exceed the vertical ring spacing on a cone at `ground_max_range`, or a far
cone breaks into one cluster per ring. `voxel_leaf` downsamples before clustering
and is the single biggest lever on frame time — see
[Performance](#performance). It also means `min_cluster_points` and
`max_cluster_points` count voxels rather than returns.

Cone acceptance, all bounds on a cluster's axis-aligned box in the car frame with
heights measured from the track surface: `max_width` (`0.50`), `min_height`
(`0.02`), `max_height` (`0.65`), `max_base_offset` (`0.25`),
`min_cluster_points` (`2`), `max_cluster_points` (`400`), `max_lateral_offset`
(`10.0`). Defaults bracket both cone sizes -- small 228 × 228 × 325 mm, big
orange 285 × 285 × 505 mm -- with margin for how sparsely a LiDAR samples them.
`min_cluster_points` and `max_cluster_points` are handed to the extraction rather
than applied afterwards, so a wall is never accumulated only to be thrown away.

Note that `min_height` is nearly zero on purpose. Past ten metres or so a cone is
caught by a single scan ring, which has no measurable height at all, so a real
height floor would throw away exactly the far detections that matter most.

Plus `enable_debug` (`true`), shared with `mirena_perception`.

The heartbeat report carries `status_state`, `frames_processed`,
`cones_last_frame` and `seconds_since_frame` under the same names
`mirena_perception` uses, and adds `nonground_last_frame`, `ground_removal_ms`,
`total_ms` and `dropped_no_transform`. `nonground_last_frame` is the one to watch
while tuning: near zero means `sensor_height` is off and the cones are being
called ground, and a large fraction of the sweep means the opposite.

## Building

```sh
colcon build --packages-select mirena_perception
```

Nothing has to be installed by hand. Patchwork++ is vendored by CMake at
configure time (pinned to `v1.4.1`), and PCL comes from apt — `libpcl-dev` and
`ros-jazzy-pcl-conversions`, both added to `docker/Dockerfile.one`, so the
devcontainer already has them. Only the `common`, `kdtree`, `search` and
`segmentation` components are asked for; pulling in all of PCL would drag in
visualization and VTK for nothing.

Configuring prints a CMake dev warning about `FLANN_ROOT` and policy CMP0144.
That comes out of PCL's own `FindFLANN.cmake`, not this package.

This used to be an Open3D dependency, for the single `ClusterDBSCAN` call.
Open3D publishes binary devel releases for x86_64 and macOS only, so on the
Jetson it meant a ~30 min source build of a library used for one function. PCL's
`EuclideanClusterExtraction` is the same grouping for this data — with
`min_cluster_points` at 2 and a fixed radius, DBSCAN's core-point rule and
single-link connectivity agree — and it is already on the image.

## Mounting

Ground removal runs on a levelled cloud: the car-from-LiDAR rotation is applied
before Patchwork++ and only the translation after. Patchwork++ wants a sensor
whose z is up. It cuts the cloud into concentric rings and gates each fitted
patch on how far its elevation sits from `-sensor_height`, with a ground
thickness of ~12 cm, so a mount tilted by `t` degrees makes the ground's z climb
by `range × tan(t)` — metres, across the ring widths it uses. The far patches
then fail the gate and the track surface is handed on as non-ground.

That fixes the tilt the extrinsic knows about. The tilt it does *not* know about
is why heights are measured against a per-cell ground map instead of a plane: one
unmodelled degree puts a flat plane 17 cm out at 10 m, which is most of a cone,
and `max_base_offset` is a 25 cm test. With the map, a wrong mount angle or a
sloping site costs accuracy in the reported position rather than every detection.

Both are worth having, but neither is a substitute for the extrinsic being right.
To check it, fit a plane to the ground returns in the car frame over a run in
which the car changes heading: a mount error is a constant pitch with near-zero
roll, while a sloping site swings with heading and shows up in roll too.

## Performance

About 28-50 ms per cloud on an Orin NX, against the 100 ms a 10 Hz sensor gives
you. The heartbeat carries the split — `level_ms`, `patchwork_ms`,
`ground_map_ms`, `band_ms`, `kdtree_ms`, `cluster_ms`, `accept_ms` — because
which stage dominates moves with the scene and with the leaf size. On the test
bag, at defaults:

| stage | mean | p90 |
| --- | --- | --- |
| `patchwork_ms` | 14.9 | 15.9 |
| `cluster_ms` | 11.0 | 16.9 |
| everything else | <2 | <3 |
| **total** | **27.9** | **34.0** |

The subscription is best-effort with a shallow queue, so what does not get
serviced is dropped rather than queued and detections never go stale.

**Density, not point count, is what costs.** Clustering does one radius search
per point and pays for every neighbour each one returns, so it scales with how
close together the points are. A 0.2° sensor puts returns about a centimetre
apart at 3 m, which is thousands inside a 30 cm radius. Downsampling to a 5 cm
voxel first took clustering from 72 ms mean / 157 ms p90 to 18 / 28, and the
whole frame from 92 / 176 to 50 / 65. It also *raised* the detection rate, from
10.7 to 11.3 cones per frame: near cones were arriving as enough returns to
exceed `max_cluster_points` and being dropped. Far cones are untouched, since at
50 m the sample spacing is already 17 cm.

Leaf size trades directly against how many voxels a distant cone survives as.
On the test bag: `0.03` → 65 ms, `0.05` → 49 ms, `0.08` → 40 ms, all within
0.1 cones per frame of each other. `0.05` leaves a 228 mm cone four to five
voxels across, which is why it is the default.

**On the Jetson itself**, the CPU governor is worth more than any of this. Under
`schedutil` a bursty 50-ms-in-every-100-ms load leaves the cores at 1.19 GHz of
their 1.98 GHz; pinning them (`jetson_clocks`, or the `performance` governor)
took the same build from 49.9 ms mean / 65.3 p90 to **27.9 / 34.0**. Nothing in
the package can do that for you — it is a machine setting.

**There is no GPU-shaped problem left here.** Ubuntu's `libpcl-dev` is built
without CUDA, so `pcl::gpu::EuclideanClusterExtraction` would mean building PCL
from source against CUDA on the Jetson — the same trap as the Open3D dependency
this package was rewritten to escape — and NVIDIA's cuPCL ships against JetPack's
CUDA 11.4/12.x rather than the 13.3 in this image. What it would buy is a share
of an 11 ms stage whose input is ~2.8 k points, against transfer and launch
overhead. Patchwork++ is the larger stage now, and upstream's own source says
they measured TBB on it and left it single-threaded, because R-VPF and the
per-patch allocations do not parallelise. Spend the Orin's GPU on the camera
pipeline.

`cluster_tolerance` has room to come down on a high-resolution sensor. It has to
exceed the sample spacing on a cone at `ground_max_range`; the M1's 0.2° gives
17 cm at 50 m, so `0.30` holds with margin, and tightening it separates cones
from what they stand next to.

## Layout

- `src/cone_detector.{hpp,cpp}` — the pipeline. No ROS, so the heuristics can be
  read and tuned without reading the node.
- `src/mirena_perception.cpp` — the ROS shell: subscriptions, TF, the
  `EntityList`, the debug clouds, the heartbeat.
