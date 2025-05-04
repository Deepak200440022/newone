import open3d as o3d
import pyvista as pv
import numpy as np


def apply_surface_reconstruction(point_cloud: o3d.geometry.PointCloud, depth=9, width=0, scale=1.1, linear_fit=True):
    """
    Apply surface reconstruction to the point cloud using Poisson surface reconstruction.

    Parameters:
    - point_cloud: The input Open3D point cloud.
    - depth: The depth of the reconstruction. Higher values give more detail (default: 9).
    - width: The width of the octree (default: 0).
    - scale: The scale of the mesh. Default is 1.1.
    - linear_fit: If True, use linear fitting. Default is True.

    Returns:
    - mesh: The reconstructed mesh.
    """
    # Check if normals are present, if not, estimate them
    if not point_cloud.has_normals():
        print("[INFO] Estimating normals...")
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
        )

    # Apply Poisson surface reconstruction
    print("[INFO] Applying Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    # Optionally, remove low-density vertices (filter out noisy parts)
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 10)
    vertices_to_keep = densities > density_threshold
    mesh.remove_vertices_by_index(np.where(~vertices_to_keep)[0])

    return mesh


def visualize_with_pyvista(mesh: o3d.geometry.TriangleMesh):
    """
    Visualize the reconstructed mesh using PyVista.

    Parameters:
    - mesh: Open3D TriangleMesh object.
    """
    # Convert the Open3D mesh to PyVista format
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Flatten faces array to match the expected PyVista format
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    faces_pv = faces_pv.flatten()

    # Create a PyVista PolyData object
    pv_mesh = pv.PolyData(vertices, faces_pv)


    # Add color from mesh's vertex colors if they exist
    if len(mesh.vertex_colors) > 0:
        colors = np.asarray(mesh.vertex_colors)
        pv_mesh.point_data["RGB"] = colors


    # Create a PyVista plotter and show the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="RGB", rgb=True, point_size=3.0)
    plotter.show()


