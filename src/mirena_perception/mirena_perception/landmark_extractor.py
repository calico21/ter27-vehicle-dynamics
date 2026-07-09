import torch
import functools
import time

def time_execution(func):
    """Decorator to measure and log execution runtime performance metrics."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Synchronize CUDA streaming lines to get accurate performance metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        result = func(self, *args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"[PROFILE] {func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    return wrapper


class LandmarkExtractor:
    """Geometric cluster operations optimized for parallelized GPU/CUDA execution."""

    @staticmethod
    def transform_points(points: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
        """Applies a 4x4 rigid transformation matrix to an (N, 3) point tensor."""
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        return torch.matmul(points, R.T) + t
    
    @time_execution
    def _project_lidar_to_cam_vectorized(pts_lidar: torch.Tensor, transform_cam2lidar: torch.Tensor, id_mask: torch.Tensor, k_tensor: torch.Tensor):
        pts_cam = LandmarkExtractor.transform_points(pts_lidar, transform_cam2lidar)

        # Filters points out behind the vehicle bumper (X < 0.1) and lens spatial plane
        front_mask = (pts_cam[:, 2] > 1.0) & (pts_lidar[:, 0] > 0.1)
        pts_cam_front = pts_cam[front_mask]
        pts_lidar_front = pts_lidar[front_mask]

        fx, fy = k_tensor[0, 0], k_tensor[1, 1]
        cx, cy = k_tensor[0, 2], k_tensor[1, 2]
        
        depth = pts_cam_front[:, 2]
        u = ((pts_cam_front[:, 0] * fx) / depth + cx).long()
        v = ((pts_cam_front[:, 1] * fy) / depth + cy).long()

        mask_h, mask_w = id_mask.shape
        valid_bounds = (u >= 0) & (u < mask_w) & (v >= 0) & (v < mask_h)
        if not valid_bounds.any():
            return None
        
        u_v, v_v = u[valid_bounds], v[valid_bounds]
        cluster_ids = id_mask[v_v, u_v]
        
        is_obj = cluster_ids > 0
        if not is_obj.any():
            return None

        return cluster_ids[is_obj], pts_cam_front[valid_bounds][is_obj], pts_lidar_front[valid_bounds][is_obj]
    
    @staticmethod
    @time_execution
    def extract_centroid_front_subcloud(
        subcloud_ids: torch.Tensor,
        pts_projected: torch.Tensor,
        pts_original: torch.Tensor,
        num_instances: int,
        min_points: int = 6,
        depth_envelope: float = 0.50
    ):
        """
        Parallelized Nearest Cluster Extraction Engine.
        Eliminates Python loops entirely by using advanced global CUDA reductions.
        """
        if subcloud_ids.numel() == 0:
            return [], [], None, None

        # Step 2: Remove background clutter
        min_depths = torch.full((num_instances + 1,), float('inf'), device=subcloud_ids.device)
        min_depths.scatter_reduce_(0, index=subcloud_ids, src=pts_projected[:, 2], reduce="amin", include_self=False)

        front_cluster_mask = pts_projected[:, 2] <= (min_depths[subcloud_ids] + depth_envelope)
        if not front_cluster_mask.any():
            return [], [], None, None

        subcloud_ids = subcloud_ids[front_cluster_mask]
        pts_projected = pts_projected[front_cluster_mask]
        pts_original = pts_original[front_cluster_mask]

        # Step 2: Calculate global instances distribution sizes natively on GPU
        subcloud_point_counts = torch.bincount(subcloud_ids, minlength=num_instances + 1)

        subcloud_enough_points_mask = subcloud_point_counts[subcloud_ids] >= min_points
        if not subcloud_enough_points_mask.any():
            return [], [], None, None
            
        subcloud_ids = subcloud_ids[subcloud_enough_points_mask]
        pts_projected = pts_projected[subcloud_enough_points_mask]
        pts_original = pts_original[subcloud_enough_points_mask]

        # Step 2: Execute Parallel Reductions for Centroids via Scatter-Adds
        unique_uids, inverse_indices = torch.unique(subcloud_ids, return_inverse=True)
        num_valid_cones = unique_uids.size(0)

        ones = torch.ones_like(subcloud_ids, dtype=torch.float32).unsqueeze(1)
        weight_sums = torch.zeros((num_valid_cones, 1), device=subcloud_ids.device)
        weight_sums.scatter_add_(0, index=inverse_indices.unsqueeze(1), src=ones)

        centroid_sums = torch.zeros((num_valid_cones, 3), device=subcloud_ids.device)
        centroid_sums.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), pts_original)

        centroids_lidar = centroid_sums / torch.clamp(weight_sums, min=1e-6)

        return centroids_lidar, unique_uids, pts_original.cpu().numpy(), subcloud_ids.cpu().numpy()
