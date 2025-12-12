import numpy as np
import open3d as o3d
import os

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


def _voxelize(mesh, name, output_dir) -> None:
    # if not opt.superquadrics:
    # mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
    # else:
      # mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, f'raw/superquadrics/{sha256}_sq_masked.ply'))
    
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, voxels_dir_name, f'{name}.ply'), vertices)
