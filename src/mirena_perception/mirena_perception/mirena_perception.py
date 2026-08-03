#!/usr/bin/env python3
import json
import time
import sys
import random
import functools
import numpy as np
import torch
import cv2  # For drawing, alphablending, and colormaps
from ultralytics import YOLO

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformException, Buffer, TransformListener
import message_filters

from mirena_common.msg import EntityList, Entity, Heartbeat
from mirena_perception.landmark_extractor import LandmarkExtractor, time_execution

# --- Global Topic Constants ---
CAMERA_INFO_SUB_TOPIC = '/sensors/camera/camera_info'
IMAGE_SUB_TOPIC = '/sensors/camera/image'
CLOUD_SUB_TOPIC = '/sensors/lidar/points'

ENTITY_LIST_PUB_TOPIC = '/perception/cones'
DEBUG_PC_PUB_TOPIC = '/perception/debug/clustered_points'
DEBUG_IMG_PUB_TOPIC = '/perception/debug/segmentation_mask'
DEBUG_PROJ_IMG_PUB_TOPIC = '/perception/debug/depth_projection'
HEALTH_PUB_TOPIC = 'health'

# The supervisor holds the mission in CONFIGURING until every pipeline node is
# beating, so this rate is what it waits on, not a diagnostic nicety.
HEARTBEAT_PERIOD_SEC = 0.1

# Activate pytorch logs
torch._logging.set_logs(graph_code=True)
torch._logging.set_logs(graph_breaks=True)


# QOS profile
sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=20
        )


class MirenaPerception(Node):
    """Synchronized YOLOv8 + LiDAR Fusion Node with complete GPU/CUDA optimization."""

    CLASS_ID_MAP = {
        0: 'cone_orange',        1: 'cone_yellow',
        2: 'cone_blue',          3: 'cone_yellow',
        4: 'cone_orange',        5: 'cone_big_orange',
        6: 'unknown_cone',       7: 'cone_big_orange',
        8: 'cone_blue',          9: 'unknown_cone'
    }

    def __init__(self):
        super().__init__('mirena_perception')
        
        self.declare_parameter('model_path', "/home/mirena/mirena_ws/src/mirena_perception/resource/best.engine")
        self.declare_parameter('yolo_conf_thresh', 0.65)
        self.declare_parameter('sync_slop_seconds', 0.1)
        self.declare_parameter('min_cluster_points', 6)
        self.declare_parameter('enable_debug', True)

        self.model = None
        self.tf_buffer = None
        self.tf_listener = None
        
        self.k_tensor = None
        self.d_tensor = None
        self.distortion_model = None
        self.cam_frame = None

        self.info_sub = None
        self.img_sub = None
        self.cloud_sub = None
        self.ts = None

        self.health_pub = None
        self.health_timer = None

        # Read by the heartbeat, written by the fusion callback.
        self.frames_processed = 0
        self.cones_last_frame = 0
        self.last_frame_at = 0.0

        self.entity_list_pub = None
        self.debug_pc_pub = None
        self.debug_img_pub = None
        self.debug_proj_img_pub = None

    def start(self) -> None:
        """Bring the node up. Raises on failure so main() can exit non-zero --
        the supervisor fails the mission when a node dies."""
        self.get_logger().info('Starting Mirena Perception Node...')

        self.model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('yolo_conf_thresh').value
        self.sync_slop = self.get_parameter('sync_slop_seconds').value

        self.yolo_device = 'cuda'
        self.device = torch.device('cuda')

        self.model = YOLO(self.model_path, task="segment")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.entity_list_pub = self.create_publisher(EntityList, ENTITY_LIST_PUB_TOPIC, 10)
        self.debug_pc_pub = self.create_publisher(PointCloud2, DEBUG_PC_PUB_TOPIC, 1)
        self.debug_img_pub = self.create_publisher(Image, DEBUG_IMG_PUB_TOPIC, 1)
        self.debug_proj_img_pub = self.create_publisher(Image, DEBUG_PROJ_IMG_PUB_TOPIC, 1)

        self.info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_SUB_TOPIC, self.info_cb, qos_profile=sensor_qos)
        self.img_sub = message_filters.Subscriber(self, Image, IMAGE_SUB_TOPIC, qos_profile=sensor_qos)
        self.cloud_sub = message_filters.Subscriber(self, PointCloud2, CLOUD_SUB_TOPIC, qos_profile=sensor_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.cloud_sub], queue_size=10, slop=self.sync_slop
        )
        self.ts.registerCallback(self.sync_cb)

        # Last, deliberately: the first heartbeat is this node telling the
        # supervisor it is up, so it must not go out before it actually is.
        # Loading the TensorRT engine above is what that wait is usually for.
        self.health_pub = self.create_publisher(Heartbeat, HEALTH_PUB_TOPIC, 10)
        self.health_timer = self.create_timer(HEARTBEAT_PERIOD_SEC, self.publish_heartbeat)

        self.get_logger().info('Mirena Perception running.')

    def publish_heartbeat(self) -> None:
        report = {
            'status_state': 'WORKING' if self.frames_processed else 'IDLE',
            'frames_processed': self.frames_processed,
            'cones_last_frame': self.cones_last_frame,
            'have_camera_info': self.k_tensor is not None,
            'seconds_since_frame':
                (time.time() - self.last_frame_at) if self.last_frame_at else -1.0,
        }
        msg = Heartbeat()
        msg.node_name = self.get_name()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.report = json.dumps(report)
        self.health_pub.publish(msg)

    def info_cb(self, msg: CameraInfo):
        if self.k_tensor is None:
            self.k_tensor = torch.tensor(msg.k, dtype=torch.float32, device=self.device).view(3, 3)
            self.d_tensor = torch.tensor(msg.d, dtype=torch.float32, device=self.device)
            self.distortion_model = msg.distortion_model.lower()
            self.cam_frame = msg.header.frame_id
    

    @time_execution
    def _project_lidar_to_cam_torch(self, pts_lidar: torch.Tensor, T_cam_lidar: torch.Tensor, id_mask: torch.Tensor):
        """Vectorized projection supporting both pinhole and high-order fisheye architectures natively on GPU."""
        # 1. Transform points into the camera optical frame
        ones = torch.ones((pts_lidar.shape[0], 1), dtype=pts_lidar.dtype, device=pts_lidar.device)
        pts_hom = torch.cat([pts_lidar, ones], dim=1)
        pts_cam = torch.matmul(pts_hom, T_cam_lidar.t())[:, :3]
        
        # 2. Distance filtering (Z-forward constraint between 0.1 meters and 25 meters)
        front_mask = (pts_cam[:, 2] > 0.1) & (pts_cam[:, 2] < 25.0)
        if not front_mask.any():
            return None
            
        pts_cam_front = pts_cam[front_mask]
        pts_lidar_front = pts_lidar[front_mask]
        
        X = pts_cam_front[:, 0]
        Y = pts_cam_front[:, 1]
        Z = pts_cam_front[:, 2]
        
        fx, fy = self.k_tensor[0, 0], self.k_tensor[1, 1]
        cx, cy = self.k_tensor[0, 2], self.k_tensor[1, 2]
        
        # 3. Apply Projection Architecture
        if self.distortion_model in ['fisheye', 'equidistant'] and self.d_tensor is not None and self.d_tensor.shape[0] >= 4:
            # Equidistant/Kannala-Brandt Projection Model
            x = X / Z
            y = Y / Z
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan(r)
            
            # Prevent NaN on division by zero at center pixel
            r_safe = torch.where(r == 0, torch.ones_like(r), r)
            
            k1, k2, k3, k4 = self.d_tensor[0], self.d_tensor[1], self.d_tensor[2], self.d_tensor[3]
            theta2 = theta ** 2
            theta4 = theta2 ** 2
            theta6 = theta4 * theta2
            theta8 = theta4 ** 2
            
            theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
            scale = torch.where(r == 0, torch.zeros_like(r), theta_d / r_safe)
            
            u = fx * (x * scale) + cx
            v = fy * (y * scale) + cy
        else:
            # Traditional Pinhole Standard model
            u = (fx * X) / Z + cx
            v = (fy * Y) / Z + cy
            
        uv = torch.stack([u, v], dim=1)
        # 4. Viewport Clamping Boundary Check
        h, w = id_mask.shape
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        
        uv_valid = uv[in_bounds]
        pts_lidar_valid = pts_lidar_front[in_bounds]
        depths_valid = pts_cam_front[in_bounds, 2]
        if uv_valid.shape[0] == 0:
            return None
            
        # Create a 3-column [u, v, depth] matrix to satisfy LandmarkExtractor expectations
        pts_cam_proj = torch.cat([uv_valid, depths_valid.unsqueeze(1)], dim=1)
        # 5. Extract instance segmentation filters
        u_idx = uv_valid[:, 0].long()

        v_idx = uv_valid[:, 1].long()
        valid_ids = id_mask[v_idx, u_idx]
        hit_mask = valid_ids > 0
        
        return (
            valid_ids[hit_mask],
            pts_cam_proj[hit_mask], # Fixed: Now passes shape (M, 3) instead of (M, 2)
            pts_lidar_valid[hit_mask],
            uv_valid,               # Sent down to debug topic ungated
            depths_valid            # Sent down to debug topic ungated
        )

    @time_execution
    def sync_cb(self, img_msg: Image, cloud_msg: PointCloud2):
        
        

        if self.k_tensor is None:
            self.get_logger().warn("Waiting for CameraInfo...", throttle_duration_sec=2.0)
            return

        debug_enabled = self.get_parameter('enable_debug').value
        min_pts = self.get_parameter('min_cluster_points').value

        # 1. Pipeline Segmentation Engine Execution
        id_mask, cone_meta, num_instances, annotated_img, raw_img = self._process_image(img_msg, draw_debug=debug_enabled)
        if id_mask is None or num_instances == 0:
            return

        # 2. Extrinsic Transformation Lookups
        lidar_frame = cloud_msg.header.frame_id
        timestamp = rclpy.time.Time.from_msg(cloud_msg.header.stamp)

        T_cam_lidar = self._get_tf_matrix(self.cam_frame, lidar_frame, timestamp)
        T_cog_lidar = self._get_tf_matrix("car/cog", lidar_frame, timestamp)
        if T_cam_lidar is None or T_cog_lidar is None:
            return

        # 3. Memory Cast Extraction
        pts_lidar = self._parse_pointcloud_cuda(cloud_msg)
        if pts_lidar.shape[0] == 0:
            return

        # 4. GPU-Accelerated Image Domain Projection & Gating 
        proj_data = self._project_lidar_to_cam_torch(pts_lidar, T_cam_lidar, id_mask)
        if proj_data is None:
            return
   
        

        valid_ids, valid_pts_cam, valid_pts_lidar, all_uv, all_depths = proj_data

        # 5. Parallelized Spatial Clustering
        centroids_lidar, unique_uids, dbg_pts, dbg_ids = LandmarkExtractor.extract_centroid_front_subcloud(
            valid_ids, valid_pts_cam, valid_pts_lidar, num_instances, min_points=min_pts
        )
        
        # 6. System Diagnostics Visualizers 
        if debug_enabled:
            self._handle_debug_publishing(
                img_msg, cloud_msg, raw_img, annotated_img, 
                dbg_pts, dbg_ids, all_uv, all_depths
            )
        if len(centroids_lidar) == 0:
            
            return

        # 7. Spatial Transforms back to Center of Gravity Reference
        centroids_cog = LandmarkExtractor.transform_points(centroids_lidar, T_cog_lidar).cpu().numpy()
        meta_list = [cone_meta.get(uid.item(), (0, 0.0)) for uid in unique_uids]
        
        self._publish_entities(centroids_cog, meta_list, cloud_msg.header.stamp)

        


    # Maps supported encodings to (channel count, OpenCV conversion code to BGR).
    # The downstream pipeline (YOLO, result.plot, debug republishing) assumes BGR.
    _ENCODING_TO_BGR = {
        'bgr8': (3, None),
        'rgb8': (3, cv2.COLOR_RGB2BGR),
        'rgba8': (4, cv2.COLOR_RGBA2BGR),
        'bgra8': (4, cv2.COLOR_BGRA2BGR),
    }

    def _msg_to_numpy(self, msg: Image):
        """Converts sensor_msgs/Image to a BGR numpy array."""
        # Validate format
        conv = self._ENCODING_TO_BGR.get(msg.encoding)
        if conv is None:
            self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
            return None

        channels, cvt_code = conv

        # Convert byte array to numpy array
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)

        if cvt_code is not None:
            img = cv2.cvtColor(img, cvt_code)

        return img


    @time_execution
    def _process_image(self, msg: Image, draw_debug: bool):
        cv_img = self._msg_to_numpy(msg)
        if cv_img is None:
            return None, {}, 0, None, None
        
        results = self.model.predict(
            source=cv_img, imgsz=640, conf=self.conf_thresh, 
            device=self.yolo_device, half=True, verbose=False, stream=True,
            retina_masks=True
        )
        result = next(results)
        annotated_frame = result.plot(boxes=False) if draw_debug else None

        if result.masks is None:
            return None, {}, 0, annotated_frame, cv_img

        masks = (result.masks.data > 0.5).long()
        num_instances = masks.shape[0]

        classes = result.boxes.cls.long().tolist()
        confs = result.boxes.conf.float().tolist()
        cone_meta = {i + 1: (classes[i], confs[i]) for i in range(num_instances)}

        ids = torch.arange(1, num_instances + 1, device=masks.device).view(num_instances, 1, 1)
        id_mask = torch.max(masks * ids, dim=0)[0]

        return id_mask, cone_meta, num_instances, annotated_frame, cv_img

    @time_execution
    def _get_tf_matrix(self, target_frame: str, source_frame: str, time: rclpy.time.Time) -> torch.Tensor:
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
        except TransformException as e:
            self.get_logger().debug(f"TF Error: {e}")
            return None
            
        q = tf.transform.rotation
        t = tf.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        
        mat = np.array([
            [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w),       t.x],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w),       t.y],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2), t.z],
            [0.0,                   0.0,                   0.0,                   1.0]
        ], dtype=np.float32)
        
        return torch.from_numpy(mat).to(self.device)

    @time_execution
    def _parse_pointcloud_cuda(self, cloud_msg: PointCloud2) -> torch.Tensor:
        fmt = '>f4' if cloud_msg.is_bigendian else '<f4'
        cloud_bytes = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        raw_grid = cloud_bytes.view(dtype=fmt).reshape(-1, cloud_msg.point_step // 4)
        
        pts_lidar = torch.from_numpy(raw_grid[:, :3]).to(self.device)
        return pts_lidar[~torch.isnan(pts_lidar).any(dim=1)]

    def _cv2_to_imgmsg(self, cv_img, encoding='bgr8'):
        """Manual conversion of numpy array to sensor_msgs/Image."""
        msg = Image()
        msg.height = cv_img.shape[0]
        msg.width = cv_img.shape[1]
        msg.encoding = encoding
        msg.step = cv_img.strides[0]
        msg.data = cv_img.tobytes()
        return msg
    @time_execution
    def _handle_debug_publishing(self, img_msg, cloud_msg, raw_img, annotated_img, dbg_pts, dbg_ids, uv_all=None, depths_all=None):

        # 1. Annotated Segmentation Mask Publisher
        if annotated_img is not None and self.debug_img_pub.get_subscription_count() > 0:
            debug_img_msg = self._cv2_to_imgmsg(annotated_img, encoding='bgr8')
            debug_img_msg.header = img_msg.header
            self.debug_img_pub.publish(debug_img_msg)

        # 2. Clustered 3D Points Publisher
        if dbg_pts is not None and len(dbg_pts) > 0 and self.debug_pc_pub.get_subscription_count() > 0:
            colors = []
            for uid in dbg_ids:
                random.seed(int(uid) * 100)
                colors.append((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
            
            debug_header = Header(stamp=cloud_msg.header.stamp, frame_id=cloud_msg.header.frame_id)
            colored_pc = self._create_colored_pointcloud(dbg_pts, np.array(colors), debug_header)
            self.debug_pc_pub.publish(colored_pc)
  
        # 3. Translucent Depth-Overlay Image Publisher 
        if uv_all is not None and depths_all is not None and self.debug_proj_img_pub.get_subscription_count() > 0:
            proj_img = raw_img.copy()
            overlay = raw_img.copy() 
            alpha = 0.55  # Opacity blend factor (0.0 = completely transparent, 1.0 = opaque)
            
            if uv_all.shape[0] > 0:
                uv_np = uv_all.cpu().numpy().astype(int)
                depths_np = depths_all.cpu().numpy()
                
                # Normalize depths to a 0-255 spectrum relative to a 20m gradient threshold
                norm_depths = np.clip(depths_np / 20.0, 0, 1) * 255
                norm_depths = norm_depths.astype(np.uint8)
                
                color_map = cv2.applyColorMap(norm_depths.reshape(-1,1), cv2.COLORMAP_JET)
                
                for i in range(len(uv_np)):
                    u, v = uv_np[i]
                    color = (int(color_map[i, 0, 0]), int(color_map[i, 0, 1]), int(color_map[i, 0, 2]))
                    cv2.circle(overlay, (u, v), 3, color, -1)
            
            # Composite overlay alpha layer onto frame canvas
            cv2.addWeighted(overlay, alpha, proj_img, 1.0 - alpha, 0, proj_img)
            
            proj_img_msg = self._cv2_to_imgmsg(proj_img, encoding='bgr8')
            proj_img_msg.header = img_msg.header
            self.debug_proj_img_pub.publish(proj_img_msg)


    def _create_colored_pointcloud(self, points_xyz: np.ndarray, colors_rgb: np.ndarray, header: Header) -> PointCloud2:
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        pc_data = [
            (x, y, z, (255 << 24) | (r << 16) | (g << 8) | b) 
            for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb)
        ]
        return point_cloud2.create_cloud(header, fields, pc_data)

    @time_execution
    def _publish_entities(self, centroids_cog: np.ndarray, meta_list: list, stamp):
        entity_list = EntityList()
        entity_list.header.frame_id = "car/cog"
        entity_list.header.stamp = stamp

        for pos_cog, (cone_type, conf) in zip(centroids_cog, meta_list):
            entity = Entity()
            entity.position.x = float(pos_cog[0])
            entity.position.y = float(pos_cog[1])
            entity.position.z = float(pos_cog[2])
            entity.type = self.CLASS_ID_MAP.get(int(cone_type), 'unknown_cone')
            entity.confidence = float(conf)
            entity_list.entities.append(entity)
            
        self.frames_processed += 1
        self.cones_last_frame = len(entity_list.entities)
        self.last_frame_at = time.time()
        self.entity_list_pub.publish(entity_list)

def main(args=None):
    rclpy.init(args=args)
    node = MirenaPerception()
    try:
        node.start()
    except Exception as e:
        node.get_logger().fatal(f"Startup failed: {e}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
