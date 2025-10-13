<h2 align="center"><b>MDP-Omni: Parameter-free Multimodal Depth Prior-based Sampling for Omnidirectional Stereo Matching</b></h2>
<h4 align="center"><b>ICCV 2025</b></h4>
<div align="center">Eunjin Son, HyungGi Jo, Wookyong Kwon, Sang Jun Lee*</div>
<br>
<p align="center"><img src="https://github.com/user-attachments/assets/f81d16ea-707f-4eb4-913e-bc7a5c4037c3"></p>

## Requirements
Install the requirements:
```bash

conda create -n MDP_omni python=3.8
conda activate MDP_omni

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Please prepare the dataset, which can be downloaded from [this link](https://rvlab.snu.ac.kr/research/omnistereo).



## TODO
 - [ ] Release training code.
 - [ ] Release inference code.
 - [ ] Release model checkpoints.
 - [ ] Release paper.

## Acknowledgements
This repository is built upon **[OmniMVS](https://github.com/hyu-cvlab/omnimvs-pytorch)**, **[RomniStereo](https://github.com/HalleyJiang/RomniStereo)**, and **[NP-CVP-MVSNet](https://github.com/NVlabs/NP-CVP-MVSNet)**.

We sincerely thank the authors for their publicly available contributions.
