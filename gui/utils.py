import numpy as np
import open3d as o3d
import os
import torch
from sklearn.decomposition import PCA


def merge_meshes(mesh_list):
    merged = o3d.geometry.TriangleMesh()
    v_offset = 0

    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles) + v_offset

        merged.vertices.extend(o3d.utility.Vector3dVector(vertices))
        merged.triangles.extend(o3d.utility.Vector3iVector(triangles))

        v_offset += len(vertices)

    return merged


def voxelgrid_to_open3d(voxels: np.ndarray, threshold=0.5):
    if len(voxels.shape) > 3:
        C, D, H, W = voxels.shape
        flat_feats = voxels.reshape(C, -1).transpose(1,0)
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(flat_feats)
        # Compute feature norm and PCA color std

        # Normalize for RGB
        reduced -= reduced.min(0)
        reduced /= reduced.max(0) + 1e-6

        # Compute norms and color std
        norms = np.linalg.norm(flat_feats, axis=1)
        color_std = np.std(reduced, axis=1)

        # Filter: active voxels with non-trivial color
        mask = (norms > threshold) & (color_std > 1e-3)

        # zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        xx, yy, zz = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        valid_coords = coords[mask]
        valid_colors = reduced[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_coords.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float32))
    else:
        coords = np.argwhere(voxels > threshold)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd


def save_voxelgrid_as_ply(voxels: np.ndarray, filename: str, threshold=0.5):
    pcd = voxelgrid_to_open3d(voxels, threshold)
    o3d.io.write_point_cloud(filename, pcd)


def voxelize_sq_francis(file_name):
    superquadric_mesh = o3d.io.read_triangle_mesh(file_name)
    vertices = np.clip(np.asarray(superquadric_mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    superquadric_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    theta = np.pi / 2
    #superquadric_mesh.rotate(R_x, center=(0, 0, 0))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        superquadric_mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    unique_points = np.unique(vertices, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    o3d.io.write_point_cloud("merged_mesh_voxelized.ply", pcd)

    zeros = np.zeros((unique_points.shape[0], 1))
    unique_points_4d = np.hstack((zeros, unique_points))  # shape [N, 4]
    unique_points_4d_torch = torch.from_numpy(unique_points_4d).to(dtype=torch.int32, device='cpu')
    my_coords_orig = unique_points_4d_torch

    coords_dense = torch.ones(1, 1, 64, 64, 64).to(device='cpu', dtype=torch.float32) * 0.0
    for i in range(my_coords_orig.shape[0]):
      x, y, z = my_coords_orig[i, 1], my_coords_orig[i, 2], my_coords_orig[i, 3]
      coords_dense[0, 0, x, y, z] = 1.0
    return coords_dense