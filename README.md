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

## Downloads
- [Synthetic datasets](https://rvlab.snu.ac.kr/research/omnistereo): OmniThings, OmniHouse, Urban
- [OmniPus](https://github.com/EunjinSon1/OmniPus)
- [Pretrained models](https://drive.google.com/drive/folders/1-huyzAzQYTdxXcs6cCmr8BEvYHj14fPv?usp=sharing)


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
