<h1 align="center">SpaceControl<br>Introducing Test-Time Spatial Control to 3D Generative Modeling</h1>

<p align="center">
              <a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1,2*</sup>,</span>
              <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>2*</sup>
              <a href="https://ianhuang.ai">Ian Huang</a><sup>2</sup>,</span>
              <a href="https://orlitany.github.io">Or Litany</a><sup>3,4</sup>,</span>
              <a href="https://people.inf.ethz.ch/pomarc">Marc Pollefeys</a><sup>1</sup>,
              <a href="https://geometry.stanford.edu/?member=guibas">Leonidas Guibas</a><sup>2</sup>
<br>
<sup>1</sup>ETH Zurich,
<sup>2</sup>Stanford University,
<sup>3</sup>Technion, 
<sup>4</sup>NVIDIA <br>
</p>

<h3 align="center"><a href="https://github.com/spacecontrol3d/spacecontrol">Code</a> | <a href="https://arxiv.org/abs/2512.05343">Paper</a> | <a href="https://spacecontrol3d.github.io">Project Page</a> </h3>
<div align="center"></div>
</p>
<p align="center">
<a href="">
<img src="https://spacecontrol3d.github.io/images/teaser.png" alt="Logo" width="100%">
</a>
</p>

Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge. Existing approaches predominantly rely on text or image prompts, which often fall short in geometric specificity: language can be ambiguous, and images are cumbersome to edit. In this work, we introduce <span style="font-size: 16px; font-weight: 600;">S</span><span style="font-size: 12px; font-weight: 700;">PACE</span><span style="font-size: 16px; font-weight: 600;">C</span><span style="font-size: 12px; font-weight: 700;">ONTROL</span>, a training-free test-time method for explicit spatial control of 3D generation. Our approach accepts a wide range of geometric inputs, from coarse primitives to detailed meshes, and integrates seamlessly with modern pre-trained generative models without requiring any additional training. A controllable parameter lets users trade off between geometric fidelity and output realism. Extensive quantitative evaluation and user studies demonstrate that <span style="font-size: 16px; font-weight: 600;">S</span><span style="font-size: 12px; font-weight: 700;">PACE</span><span style="font-size: 16px; font-weight: 600;">C</span><span style="font-size: 12px; font-weight: 700;">ONTROL</span> outperforms both training-based and optimization-based baselines in geometric faithfulness while preserving high visual quality. Finally, we present an interactive user interface that enables online editing of superquadrics for direct conversion into textured 3D assets, facilitating practical deployment in creative workflows.

***Check out our [Project Page](https://spacecontrol3d.github.io/) for more videos and interactive demos!***


## 📦 Installation

1. Clone the repository:
  ```sh
  git clone git@github.com:spacecontrol3d/spacecontrol.git
  cd spacecontrol
  ```
2. Setup the environment:
  The code has been test with **CUDA 12.8** (see `nvcc --version`) on an NVIDIA 3090 with `torch 2.8.0+cu128`

  ```sh
  conda create -n spacecontrol python=3.10 -y
  conda activate spacecontrol
  
  # instructions for your setup: https://pytorch.org/get-started/locally/
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

  pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers psutil viser tensorboard pandas lpips
  pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

  pip install xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128
  pip install flash-attn --no-build-isolation
  pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
  pip install spconv-cu120

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

## 💡 Usage
To start the web-based interactive demo:
```sh
python gui/gui_text_image.py
```

## 🙏 Acknowledgments
We thank the authors of [TRELLIS](https://github.com/microsoft/TRELLIS) for their excellent work and for making their code publicly available. We also gratefully acknowledge NVIDIA for their academic compute grant, which enabled the development of this method; these contributions were instrumental to the project.

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
