import open3d as o3d

def refine_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=None, estimate_normals=True, normal_radius=0.1):
    """
    Refines an Open3D point cloud by removing outliers, optionally downsampling, and estimating normals.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Input point cloud to refine.
    nb_neighbors : int
        Number of neighbors to analyze for outlier removal.
    std_ratio : float
        Standard deviation multiplier for statistical outlier removal.
    voxel_size : float or None
        If provided, downsample the point cloud with this voxel size.
    estimate_normals : bool
        Whether to estimate normals after cleaning.
    normal_radius : float
        Search radius for normal estimation.

    Returns:
    --------
    refined_pcd : open3d.geometry.PointCloud
        The refined (cleaned, optional downsampled, normal-estimated) point cloud.
    """
    print("[INFO] Removing statistical outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    refined_pcd = pcd.select_by_index(ind)

    print(f"[INFO] Kept {len(refined_pcd.points)} points after outlier removal.")

    if voxel_size is not None:
        print(f"[INFO] Downsampling with voxel size {voxel_size}...")
        refined_pcd = refined_pcd.voxel_down_sample(voxel_size=voxel_size)

    if estimate_normals:
        print("[INFO] Estimating normals...")
        refined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.3, max_nn=30))

        if refined_pcd.has_normals():
            print("[INFO] Orienting normals...")
            refined_pcd.orient_normals_consistent_tangent_plane(30)
        else:
            print("[WARNING] Normals were not computed. Skipping orientation.")

    return refined_pcd
