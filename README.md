## 📦 Installation

conda deactivate
conda env remove -n spacecontrol -y

python -c "import torch; print(torch.__version__)"

1. Clone the repository:
  ```sh
  git clone git@github.com:spacecontrol3d/spacecontrol.git
  cd spacecontrol
  ```

The code has been test with CUDA 12.8 (see output of `nvcc --version`) on an NVIDIA 3090 with `torch 2.8.0+cu128`

2. Setup the environment:
  ```sh
  conda create -n spacecontrol python=3.10 -y
  conda activate spacecontrol
  
  # find the right installation instructions for you setup here: https://pytorch.org/get-started/locally/
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

  pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers psutil
  pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

  pip install tensorboard pandas lpips

  pip install xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128
  pip install flash-attn --no-build-isolation
  pip install kaolin==0.18.0pip https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

  pip install spconv
  pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

  mkdir -p /tmp/extensions
  git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
  pip install /tmp/extensions/nvdiffrast --no-build-isolation

  git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
  pip install /tmp/extensions/diffoctreerast --no-build-isolation

  git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
  pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

  cp -r extensions/vox2seq /tmp/extensions/vox2seq
  pip install /tmp/extensions/vox2seq --no-build-isolation
  ```

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{fedele2025spacecontrol,
  title   = {{SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling}},
  author  = {Elisabetta Fedele, Francis Engelmann, Ian Huang, Or Litany, Marc Pollefeys, Leonidas Guibas},
  journal = {arXiv preprint arXiv:2512.05343},
  year    = {2025}
}
```