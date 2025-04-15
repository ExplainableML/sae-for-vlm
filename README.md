## 
<h1 align="center">Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models</h1>

<div align="center">
<a href="https://www.eml-munich.de/people/mateusz-pach">Mateusz Pach</a>,
<a href="https://www.eml-munich.de/people/shyamgopal-karthik">Shyamgopal Karthik</a>,
<a href="https://www.eml-munich.de/people/quentin-bouniot">Quentin Bouniot</a>,
<a href="https://www.eml-munich.de/people/serge-belongie">Serge Belongie</a>,
<a href="https://www.eml-munich.de/people/zeynep-akata">Zeynep Akata</a>
<br>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.02821)
</div>

<h3 align="center">Abstract</h3>

<p align="justify">
Sparse Autoencoders (SAEs) have recently been shown to enhance interpretability and steerability in Large Language Models (LLMs). In this work, we extend the application of SAEs to Vision-Language Models (VLMs), such as CLIP, and introduce a comprehensive framework for evaluating monosemanticity in vision representations. Our experimental results reveal that SAEs trained on VLMs significantly enhance the monosemanticity of individual neurons while also exhibiting hierarchical representations that align well with expert-defined structures (e.g., iNaturalist taxonomy). Most notably, we demonstrate that applying SAEs to intervene on a CLIP vision encoder, directly steer output from multimodal LLMs (e.g., LLaVA) without any modifications to the underlying model. These findings emphasize the practicality and efficacy of SAEs as an unsupervised approach for enhancing both the interpretability and control of VLMs.
</p>
<br>
<div align="center">
    <img src="assets/teaser.svg" alt="Teaser" width="400">
</div>

---
### Setup
Install required PIP packages.
```bash
pip install -r requirements.txt
```
Download following datasets:
* ImageNet (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)
* INaturalist 2021 (https://github.com/visipedia/inat_comp/tree/master/2021)

Export paths to dataset directories. The directories should contain `train/` and `val/` subdirectories.
```bash
export IMAGENET_PATH="<path_to_imagenet>"
export INAT_PATH="<path_to_inaturalist>"
```
Code was run using Python version 3.11.10.
### Running Experiments
The commands required to reproduce the results are organized into scripts located in the `scripts/` directory:
* `monosemanticity_score.sh` computes the Monosemanticity Score (MS) for specified SAEs, layers, models, and image encoders.
* `matryoshka_hierarchy.sh` analyzes the hierarchical structure that emerges in Matryoshka SAEs.
* `mllm_steering.sh` enables experimentation with steering LLaVA using an SAE built on top of the vision encoder.

We use the implementation of sparse autoencoders available at https://github.com/saprmarks/dictionary_learning.
### Citation
```bibtex
@article{pach2025sparse,
  title={Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models}, 
  author={Mateusz Pach and Shyamgopal Karthik and Quentin Bouniot and Serge Belongie and Zeynep Akata},
  journal={arXiv preprint arXiv:2504.02821},
  year={2025}
}
```
