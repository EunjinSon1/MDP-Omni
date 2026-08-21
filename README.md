<h2 align="center"><b>MDP-Omni and OmniPus</b></h2>

> **MDP-Omni: Parameter-free Multimodal Depth Prior-based Sampling for Omnidirectional Stereo Matching, [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Son_MDP-Omni_Parameter-free_Multimodal_Depth_Prior-based_Sampling_for_Omnidirectional_Stereo_Matching_ICCV_2025_paper.pdf)**
> 
> Eunjin Son, HyungGi Jo, Wookyong Kwon, Sang Jun Lee*

> **(Under review) Towards Real-World Omnidirectional Stereo Matching with Multimodal Depth Prior-Based Sampling**
> 
> Eunjin Son, HyungGi Jo, Wookyong Kwon, Sang Jun Lee*

<img width="1696" height="575" alt="son7" src="https://github.com/user-attachments/assets/f33ca0bb-b328-4960-bd93-04be79887885" />


## Requirements
Install the requirements:
```bash

conda create -n MDP_omni python=3.8
conda activate MDP_omni

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Downloads
- [Synthetic datasets](https://rvlab.snu.ac.kr/research/omnistereo): OmniThings, OmniHouse, Urban
- [OmniPus](https://github.com/EunjinSon1/OmniPus)
- [Pretrained models](https://drive.google.com/drive/folders/1-huyzAzQYTdxXcs6cCmr8BEvYHj14fPv?usp=sharing)

## OmniPus
We introduce the first real-world dataset, OmniPus, for multi-view fisheye-based omnidirectional stereo matching.
It contains 24,882 samples across 33 sequences collected at three university campuses: Jeonbuk National University (OmniPus-A), Jeonju University (OmniPus-B), and Wonkwang University (OmniPus-C).
Each sample includes 960×540 multi-view RGB images and a 640×160 depth map captured using NileCam21 cameras and an Ouster OS1 LiDAR.

<img src="https://github.com/user-attachments/assets/ea4958b0-b36c-46da-92a2-3ff957fc9106" width="60%">


For privacy reasons, OmniPus is available upon request via [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSf0_dXAtZGoctc5Sb8huWU7ag_nuWyF30FNMlro2GSMXmDwng/viewform?usp=publish-editor).

OmniPus consists of three subsets, OmniPus-A, OmniPus-B, and OmniPus-C, with the following folder structure:
```bash
OmniPus/
├──calibration
    ├── cam.yaml             # Calibration parameters in .yaml format
    └── mask.png             # Masks for fisheye camera 
├──omnipusa/                 # OmniPus-A
    └──case/
       ├── resize_cam1       # Resized front camera RGB images
       ├── resize_cam2       # Resized right camera RGB images
       ├── resize_cam3       # Resized back camera RGB images
       ├── resize_cam4       # Resized left camera RGB images
       └── omnidepth_gt      # Depth maps in .tiff format saved as inverse depth
├──omnipusb/                 # OmniPus-B
└──omnipusc/                 # OmniPus-C
``` 


## Training
Train on OmniThings:
```bash
python train.py --dbname omnithings
```
Finetune on OmniHouse and Sunny:
```bash
python train.py --dbname omnihouse sunny --total_epochs 15 --lr 0.000354 --pretrain_ckpt ./checkpoints/<checkpoint_file>.pth
```
Train on OmniPus:
```bash
python train.py --dbname <dbname> --total_epochs 15 --wdecay 0.1 --sigmoid_param 5 --phi_deg 42.4
# dbname can be omnipusa, omnipusb, omnipusc
```

## Evaluation
Evaluate on the synthetic dataset:
```bash
python eval.py --dbname <dbname> --restore_ckpt ./checkpoints/<checkpoint_file>.pth --save_misc
```
Evaluate on OmniPus:
```bash
python eval.py --dbname <dbname> --phi_deg 42.4 --restore_ckpt ./checkpoints/<checkpoint_file>.pth --save_misc
# To evaluate in metric scale, add --eval_metric
```

## Acknowledgements
This repository is built upon **[OmniMVS](https://github.com/hyu-cvlab/omnimvs-pytorch)**, **[RomniStereo](https://github.com/HalleyJiang/RomniStereo)**, and **[NP-CVP-MVSNet](https://github.com/NVlabs/NP-CVP-MVSNet)**.

We sincerely thank the authors for their publicly available contributions.
