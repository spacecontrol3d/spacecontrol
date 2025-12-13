import numpy as np
from plyfile import (PlyData, PlyElement)
import os
import viser
import viser.transforms as tf
import time
import open3d as o3d
from utils import merge_meshes
import copy
from PIL import Image
from io import BytesIO


import sys
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils

RESOLUTION = 32
pipeline = None
generated_mesh = None
steps = 12
cfg_strength = 7.5

scene_elements = {}
gui_elements = {}
superquadrics = {}
active_superquadric = -1
active_template_id = 1

server = viser.ViserServer(up_axis=2)
server.scene.set_up_direction([0.0, 0.0, 1.0])
server.scene.set_environment_map('studio', background=False, environment_intensity=0.5)
point_light = server.scene.add_light_ambient('light_a', color=(255, 255, 255), intensity=10000.0)

server.gui.configure_theme(dark_mode=True)
@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
  client.camera.position = (0.8, -0.8, 0.8)
  client.camera.look_at = (0., 0., 0.)


def get_mesh_from_sq_param(scale, rot3x3, position, exps, N):
    vertices, triangles = add_superquadric_compact_rot_mat(scale, exps, position, rot3x3, N)
    return vertices, triangles


def add_superquadric_compact_rot_mat(
        scalings: np.array=np.array([1.0, 1.0, 1.0]),
        exponents: np.array=np.array([2.0, 2.0, 2.0]),
        translation: np.array=np.array([0.0, 0.0, 0.0]),
        rotation: np.array=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0,1.0]]),
        resolution: int=10,
        visible: bool=True):
        """Adds a superqiadroc mesh to the scene."""

        def create_superquadric_mesh(A, B, C, e1, e2, N):
            def f(o, m):
                return np.sign(np.sin(o)) * np.abs(np.sin(o))**m
            def g(o, m):
                return np.sign(np.cos(o)) * np.abs(np.cos(o))**m
            u = np.linspace(-np.pi, np.pi, N, endpoint=True)
            v = np.linspace(-np.pi/2.0, np.pi/2.0, N, endpoint=True)
            u = np.tile(u, N)
            v = (np.repeat(v, N))
            if np.linalg.det(rotation) < 0:
                u = u[::-1]
            triangles = []

            x = A * g(v, e1) * g(u, e2)
            y = B * g(v, e1) * f(u, e2)
            z = C * f(v, e1)
            # Set poles to zero to account for numerical instabilities in f and g due to ** operator
            x[:N] = 0.0
            x[-N:] = 0.0
            vertices =  np.concatenate([np.expand_dims(x, 1),
                                        np.expand_dims(y, 1),
                                        np.expand_dims(z, 1)], axis=1)
            vertices =  (rotation @ vertices.T).T +translation  # TODO verify left or right apply rotation

            triangles = []
            for i in range(N-1):
                for j in range(N-1):
                    triangles.append([i*N+j, i*N+j+1, (i+1)*N+j])
                    triangles.append([(i+1)*N+j, i*N+j+1, (i+1)*N+(j+1)])
            # Connect first and last vertex in each row
            for i in range(N - 1):
                triangles.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
                triangles.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])

            triangles.append([(N-1)*N+(N-1), (N-1)*N, (N-1)])
            triangles.append([(N-1), (N-1)*N, 0])

            return vertices, triangles


        vertices, triangles = create_superquadric_mesh(scalings[0], scalings[1], scalings[2],
                                                    exponents[0], exponents[1],
                                                    resolution)
        return vertices, triangles


def export_mesh(vertices, faces_idx, vertex_color, face_color, filename):
    if vertex_color is not None:
        vertex = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(vertices.shape[0]):
            vertex[i] = (vertices[i][0], vertices[i][1], vertices[i][2],vertex_color[i,0],vertex_color[i,1],vertex_color[i,2])
    else:
        vertex = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(vertices.shape[0]):
            vertex[i] = (vertices[i][0], vertices[i][1], vertices[i][2])
    if face_color is not None:
        faces = np.zeros(faces_idx.shape[0], dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(faces_idx.shape[0]):
            faces[i] = ([faces_idx[i][0], faces_idx[i][1], faces_idx[i][2]],face_color[i,0],face_color[i,1],face_color[i,2])
    else:
        faces = np.zeros(faces_idx.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
        for i in range(faces_idx.shape[0]):
            faces[i] = ([faces_idx[i][0], faces_idx[i][1], faces_idx[i][2]])

    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices']),
                       PlyElement.describe(faces, 'face')],text=True)
    ply_out.write(filename)
    return ply_out


def save_assets(input_sq, text_prompt, generated_glb, t0):
  global active_template_id
  timestamp = time.strftime("%Y%m%d-%H%M%S")
  output_dir = f'generated_assets/{timestamp}_{text_prompt.replace(" ", "_")}_{t0}'
  os.makedirs(output_dir, exist_ok=True)
  generated_glb.export(f"{output_dir}/generated.glb")
  o3d.io.write_triangle_mesh(f"{output_dir}/input_mesh.ply", input_sq)
  with open(f"{output_dir}/text_prompt.txt", "w") as f:
      f.write(text_prompt)


def generate(superquadrics, text_prompt_handle, t0_idx, image_control=False) -> None:
  print('generate')
  suffix_cur = '_with_image' if image_control else ''
  suffix_other = '' if image_control else '_with_image'
  old_disabled = gui_elements[f'generate_button{suffix_other}'].disabled
  gui_elements[f'generate_button{suffix_other}'].disabled = True
  gui_elements[f'generate_button{suffix_cur}'].label = f"Generating{suffix_cur.replace('_', ' ')}..."
  gui_elements[f'generate_button{suffix_cur}'].icon = viser.Icon.LOADER
  gui_elements[f'generate_button{suffix_cur}'].color = 'orange'
  meshes = []
  for superquadric_id in superquadrics.keys():
    vertices, triangles = add_superquadric_compact_rot_mat(
      superquadrics[superquadric_id]['scale'],
      superquadrics[superquadric_id]['shape'],
      superquadrics[superquadric_id]['translation'],
      superquadrics[superquadric_id]['rotation'], resolution=100)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    meshes.append(mesh)
  merged_mesh = merge_meshes(meshes)
  aabb = np.stack([np.asarray(merged_mesh.vertices).min(0), np.asarray(merged_mesh.vertices).max(0)])
  center = (aabb[0] + aabb[1]) / 2
  scale = 1/((aabb[1] - aabb[0]).max())

  merged_mesh.translate(-center)
  merged_mesh.scale(scale, (0,0,0))
  spatial_control_mesh_path = "gui/spatial_control_mesh.ply"
  o3d.io.write_triangle_mesh(spatial_control_mesh_path, merged_mesh)

  global pipeline
  if pipeline is None:
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

  text_prompt = text_prompt_handle.value
  image_prompt = None
  if image_control and len(image_prompt_handle.value.name) > 0:
    image_prompt = Image.open(BytesIO(image_prompt_handle.value.content))
  
  outputs = pipeline.run(text_prompt, image_prompt, seed=1, sparse_structure_sampler_params={
        "steps": steps,
        "cfg_strength": cfg_strength,
        "t0_idx_value": t0_idx.value,
        "spatial_control_mesh_path": spatial_control_mesh_path,
    })

  # video = render_utils.render_video(outputs['gaussian'][0], bg_color=(255, 255, 255), r=2)['color']
  # imageio.mimsave("sample_gs.mp4", video, fps=30)

  glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
  )
  glb.export("sample.ply")
  glb.export("sample.glb")

  glb.apply_scale(1/scale)
  glb.apply_translation(center) # bring to original scale and position before saving
  save_assets(input_sq=merged_mesh, text_prompt=text_prompt, generated_glb=glb, t0=t0_idx.value)

  global generated_mesh
  generated_mesh = server.scene.add_mesh_trimesh("generated_mesh", mesh=glb, visible=True)
  toggle_sq_mesh()
  toggle_sq_mesh()

  gui_elements[f'generate_button{suffix_other}'].disabled = old_disabled
  gui_elements[f'generate_button{suffix_cur}'].label = f"Generate{suffix_cur.replace('_', ' ')}"
  gui_elements[f'generate_button{suffix_cur}'].icon = viser.Icon.PLAYER_PLAY
  gui_elements[f'generate_button{suffix_cur}'].color = 'green'


def get_all_templates() -> list:
  return sorted([f.split('_')[0] for f in os.listdir('gui/superquadrics/') if f.endswith('_sq.npz')])


def handle_upload_image(event):
  global gui_elements
  with gui_elements['folder_image_conditioning']:
    try:
      gui_elements['image_prompt'].remove()
    except:
       pass
    gui_elements['image_prompt'] = server.gui.add_image(np.array(Image.open(BytesIO(image_prompt_handle.value.content))), order = 11)
    gui_elements['generate_button_with_image'].disabled = False


def setup_gui(server, superquadrics: dict) -> None:
  global gui_elements
  global scene_elements
  global active_template_id
  global active_superquadric

  gui_elements = {}
  active_superquadric = -1
  server.gui.reset()
  server.scene.reset()

  scene_elements = {}
  server.gui.set_panel_label("Superquadrics")
  t0_idx = server.gui.add_slider(f"Control strength (t0)", order=3, min=0, max=steps, step=1.0, initial_value=6.0, marks=((0, "0"), (steps // 3, f"{steps // 3}"), (2 * steps // 3, f"{2 * steps // 3}")),)
  text_prompt = server.gui.add_text("Text prompt", "chair", disabled=False, order=4)
  global image_prompt_handle 
  
  select_template_dropdown = server.gui.add_dropdown(label="Object Template",
                          options=[str(i) for i in range(len(get_all_templates()))],
                          order=0, initial_value=str(active_template_id))
  select_template_dropdown.on_update(lambda _: select_template(int(select_template_dropdown.value)))

  for id, superquadric in superquadrics.items():
      gui_elements_per_sq = {}
      gui_elements_per_sq['folder'] = server.gui.add_folder(
        f'Superquadric {id}', order=1, expand_by_default=True, visible=False)
      with gui_elements_per_sq[f'folder']:
        gui_elements_per_sq['shape_1'] = server.gui.add_slider(f"Shape 1", min=0, max=2, step=0.01, initial_value=superquadric['shape'][0], marks=((0, "0"), (1, "1"), (2, "2")),)
        gui_elements_per_sq['shape_2'] = server.gui.add_slider(f"Shape 2", min=0, max=2, step=0.01, initial_value=superquadric['shape'][1], marks=((0, "0"), (1, "1"), (2, "2")),)
        gui_elements_per_sq['scale_x'] = server.gui.add_slider(f"Scale X", min=0, max=1, step=0.002, initial_value=superquadric['scale'][0], marks=((0, "0"), (1, "1"), (2, "2")),)
        gui_elements_per_sq['scale_y'] = server.gui.add_slider(f"Scale Y", min=0, max=1, step=0.002, initial_value=superquadric['scale'][1], marks=((0, "0"), (1, "1"), (2, "2")),)
        gui_elements_per_sq['scale_z'] = server.gui.add_slider(f"Scale Z", min=0, max=1, step=0.002, initial_value=superquadric['scale'][2], marks=((0, "0"), (1, "1"), (2, "2")),)

        for k in gui_elements_per_sq.keys():
          try:
            gui_elements_per_sq[k].on_update(lambda _: update_sq(superquadrics, active_superquadric, resolution=RESOLUTION))
          except:
            pass
        gui_elements_per_sq['duplicate_button'] = server.gui.add_button("Duplicate", color='blue', icon=viser.Icon.COPY)
        gui_elements_per_sq['duplicate_button'].on_click(lambda _: duplicate_active_superquadric())
        gui_elements_per_sq['delete_button'] = server.gui.add_button("Delete", color='red', icon=viser.Icon.CROSS)
        gui_elements_per_sq['delete_button'].on_click(lambda _: delete_active_superquadric())
      gui_elements[f'sq_{id}'] = gui_elements_per_sq

  gui_elements['generate_button'] = server.gui.add_button("Generate", color='green', icon=viser.Icon.PLAYER_PLAY, order=5)
  gui_elements['generate_button'].on_click(lambda _: generate(superquadrics, text_prompt, t0_idx))
  
  server.gui.add_folder("", expand_by_default=False, order=6)

  gui_elements['folder_image_conditioning'] = server.gui.add_folder("Optional image prompt (texture only)", order=7, expand_by_default=True)
  with gui_elements['folder_image_conditioning']:
    image_prompt_handle = server.gui.add_upload_button("Select image prompt", color = 'gray', order=10)
    image_prompt_handle.on_upload(handle_upload_image)
    gui_elements['generate_button_with_image'] = server.gui.add_button("Generate with image", disabled = True, color='green', icon=viser.Icon.PLAYER_PLAY, order = 12)
    gui_elements['generate_button_with_image'].on_click(lambda _: generate(superquadrics, text_prompt, t0_idx, True))
  toggle_button = server.gui.add_button("Toggle", color='gray', order = 100)
  toggle_button.on_click(lambda _: toggle_sq_mesh())

  return gui_elements


def duplicate_active_superquadric() -> None:
  global superquadrics
  global scene_elements
  global gui_elements
  global active_superquadric  
  print('Duplicating SQ', active_superquadric)
  new_superquadric_id = max(superquadrics.keys()) + 1
  superquadrics[new_superquadric_id] = copy.deepcopy(superquadrics[active_superquadric])
  superquadrics[new_superquadric_id]['translation'] += np.array([0.02, 0.02, 0.02])
  copied_superquadric_id = active_superquadric
  gui_elements = setup_gui(server, superquadrics)
  for superquadric_id in superquadrics.keys():
    add_superquadric(superquadrics, superquadric_id, gui_elements, resolution=RESOLUTION)
  active_superquadric = copied_superquadric_id
  sq_on_click(new_superquadric_id)


def delete_active_superquadric() -> None:
  global superquadrics
  global scene_elements
  global gui_elements
  global active_superquadric

  print('Deleting SQ', active_superquadric)
  if active_superquadric == -1:
     return
  superquadrics.pop(active_superquadric)
  scene_elements[f'sq_{active_superquadric}'].remove()
  scene_elements[f'sqc_{active_superquadric}'].remove()
  gui_elements[f'sq_{active_superquadric}']['folder'].visible = False
  del scene_elements[f'sq_{active_superquadric}']
  del scene_elements[f'sqc_{active_superquadric}']
  active_superquadric = -1


def toggle_sq_mesh() -> None:
  generated_mesh.visible = not generated_mesh.visible
  if generated_mesh.visible:
    server.scene.set_environment_map('studio', background=False, environment_intensity=2.0)
  else:
    server.scene.set_environment_map('studio', background=False, environment_intensity=0.5)

  for key in scene_elements.keys():
    if key.startswith('sq_'):
      scene_elements[key].visible = not generated_mesh.visible
  if active_superquadric != -1:
    scene_elements[f'sqc_{active_superquadric}'].visible = not generated_mesh.visible


def update_sq(superquadrics, superquadric_id, resolution) -> None:
  superquadrics[superquadric_id]['shape'][0] = gui_elements[f'sq_{superquadric_id}']['shape_1'].value
  superquadrics[superquadric_id]['shape'][1] = gui_elements[f'sq_{superquadric_id}']['shape_2'].value
  superquadrics[superquadric_id]['scale'][0] = gui_elements[f'sq_{superquadric_id}']['scale_x'].value
  superquadrics[superquadric_id]['scale'][1] = gui_elements[f'sq_{superquadric_id}']['scale_y'].value
  superquadrics[superquadric_id]['scale'][2] = gui_elements[f'sq_{superquadric_id}']['scale_z'].value
  add_superquadric(superquadrics, superquadric_id, gui_elements, resolution)


def add_superquadric(superquadrics: dict, superquadric_id: int, gui_elements: dict, resolution) -> None:
    global scene_elements
    def create_mesh(superquadric_id, resolution) -> None:
      vertices, triangles = add_superquadric_compact_rot_mat(
          superquadrics[superquadric_id]['scale'],
          superquadrics[superquadric_id]['shape'],
          superquadrics[superquadric_id]['translation'],
          superquadrics[superquadric_id]['rotation'],
          resolution)

      scene_elements[f'sq_{superquadric_id}'] = server.scene.add_mesh_simple(
          name=f"/sq/{superquadric_id}",
          vertices=vertices,
          color=superquadrics[superquadric_id]['color'],
          faces=np.array(triangles),
      )

      scene_elements[f'sqc_{superquadric_id}'] = server.scene.add_transform_controls(
         f'sqc_{superquadric_id}', scale=0.2, line_width=2.5, fixed=False, visible=superquadric_id==active_superquadric,
         active_axes=[True, True, True], depth_test=False,
         position=superquadrics[superquadric_id]['translation'],
         wxyz=tf.SO3.from_matrix(superquadrics[superquadric_id]['rotation']).wxyz)
      
      @scene_elements[f'sqc_{superquadric_id}'].on_update
      def _(_) -> None:
          superquadrics[superquadric_id]['translation'] = scene_elements[f'sqc_{superquadric_id}'].position
          superquadrics[superquadric_id]['rotation'] = tf.SO3.as_matrix(scene_elements[f'sqc_{superquadric_id}'])
          update_sq(superquadrics, superquadric_id, resolution=RESOLUTION)


      if active_superquadric != superquadric_id:
        ha = scene_elements[f'sq_{superquadric_id}'].on_click(lambda _: sq_on_click(superquadric_id))
      # print(scene_elements[f'sq_{superquadric_id}']._impl.click_cb)

    create_mesh(superquadric_id, resolution)


def sq_on_click(superquadric_id):
  global active_superquadric
  print(f"Clicked on superquadric {superquadric_id}")
  print(f"Active superquadric {active_superquadric}")
  print(f"Len superquadric {len(superquadrics)}")
  if active_superquadric != -1:
    # at this point, active_superquadric is the one that was clicked previoulsy
    scene_elements[f'sq_{active_superquadric}'].on_click(lambda _: sq_on_click(active_superquadric))
    superquadrics[active_superquadric]['color'] = [90, 200, 255]  # reset color of previously selected superquadric  [90, 200, 255]
  active_superquadric = superquadric_id
  scene_elements[f'sq_{active_superquadric}'].remove_click_callback('all')
  superquadrics[active_superquadric]['color'] = [255, 0, 255]  # update color of newly selected superquadric
  for i in superquadrics.keys():
      gui_elements[f'sq_{i}']['folder'].visible = i == active_superquadric
      scene_elements[f'sqc_{i}'].visible = i == active_superquadric
  for i in superquadrics.keys():
      update_sq(superquadrics, i, resolution=RESOLUTION)


def load_superquadric_from_file(file_path: str) -> list:
  par_dict = np.load(file_path)
  scale = par_dict['scales']        # 3 (3x1 vector)
  rotate = par_dict['rotations']    # 3 (3x3 rotation matrix)
  shapes = par_dict['shapes']       # 2 (2x1 vector)
  trans = par_dict['translations']  # 3 (3x1 vector)
  num_el = scale.shape[0]           # number of superquadrics

  superquadrics = {}
  for k in range(num_el):
    superquadric_dict = {}
    superquadric_dict['scale'] = scale[k, :]
    superquadric_dict['shape'] = shapes[k]
    superquadric_dict['rotation'] = rotate[k, :]
    superquadric_dict['translation'] = trans[k, :]
    superquadric_dict['color'] = [90, 200, 255]
    superquadrics[k] = superquadric_dict
  return superquadrics


def select_template(template_id: int) -> None:
  global active_template_id
  global superquadrics
  active_template_id = template_id
  input_path = os.path.join('gui/superquadrics/', f'{get_all_templates()[template_id]}_sq.npz')

  print(f"Loading superquadrics from {input_path}")
  superquadrics = load_superquadric_from_file(input_path)
  gui_elements = setup_gui(server, superquadrics)
  for superquadric_id in range(len(superquadrics)):
    add_superquadric(superquadrics, superquadric_id, gui_elements, resolution=RESOLUTION)


def main():
  select_template(0)
  while True:
      time.sleep(10.0)

if __name__ == '__main__':
  main()