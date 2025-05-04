import cv2
import numpy as np
import torch
import open3d as o3d
import pyvista as pv
from exif_extractor import  extract_camera_intrinsics
from pre_processing import refine_point_cloud
import mesh_generatation

# === Load depth map (grayscale float) ===
depth_map = cv2.imread("/home/deepak/PycharmProjects/MiniProject/predicted_depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

# # === Optional inversion if necessary ===
if depth_map.max() > depth_map.min():
    depth_map = depth_map.max() - depth_map  # invert depth if needed
# === Load RGB image ===
rgb_image = cv2.imread("/home/deepak/Downloads/morskie-oko-tatry.jpg", cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # BGR → RGB
rgb_image = rgb_image.astype(np.float32) / 255.0

height, width = depth_map.shape

# === Convert to PyTorch tensors ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
depth_torch = torch.from_numpy(depth_map).to(device)
rgb_torch = torch.from_numpy(rgb_image).to(device)


intrinsics = extract_camera_intrinsics("depth_maps/example.jpg")
fx = intrinsics['fx']
fy = intrinsics['fy']
cx = intrinsics['cx']
cy = intrinsics['cy']


# === Generate (u,v) pixel grid ===
u = torch.arange(0, width, device=device).float()
v = torch.arange(0, height, device=device).float()
grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')

# === Compute 3D coordinates ===
Z = depth_torch
X = (grid_u - cx) * Z / fx
Y = (grid_v - cy) * Z / fy

# === Flatten everything ===
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
color_flat = rgb_torch.view(-1, 3)

# === Filter invalid points ===
valid = (Z_flat > 0) & (~torch.isnan(Z_flat))

points = torch.stack((X_flat[valid], Y_flat[valid], -Z_flat[valid]), dim=1).cpu().numpy()
colors = color_flat[valid].cpu().numpy()

# === Create Open3D point cloud ===
pcd = o3d.geometry.PointCloud()
pcd = refine_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=0.01, estimate_normals=True)
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)




# === Convert Open3D → numpy for PyVista ===
points_np = np.asarray(pcd.points)
colors_np = (np.asarray(pcd.colors) * 255).astype(np.uint8)

# === Create PyVista PolyData ===
point_cloud = pv.PolyData(points_np)
point_cloud["RGB"] = colors_np

from pathlib import Path

# Save the point cloud to a file using pathlib
output_path = Path("output_point_cloud.ply")
o3d.io.write_point_cloud(output_path, pcd)



mesh = mesh_generatation.apply_surface_reconstruction(pcd)

# === Step 3: Visualize with PyVista ===
print("[INFO] Visualizing the reconstructed mesh with PyVista...")
mesh_generatation.visualize_with_pyvista(mesh)

# === Visualize with PyVista ===
plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="RGB", rgb=True, point_size=3.0)
plotter.show()
