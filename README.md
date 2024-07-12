# CF-Diff
This is the pytorch implementation of our paper at SIGIR 2024:
> [Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity](https://arxiv.org/pdf/2404.14240)
> 
> Hou, Yu and Park, Jin-Duk and Shin, Won-Yong

The implementation of diffusion model and evaluation parts are referred to [DiffRec](https://github.com/YiyanXu/DiffRec/tree/main). Thank you for this contribution.
## Environment
- Anaconda 3
- python 3.8.17
- pytorch 1.13.1
- numpy 1.24.3
- math

## Usage
### Data
The user-item interactions, train/valid/test, are in './datasets' folder. "sec_hop_inters_ML_1M.pt" contains the information of second-hop user-item interactions and "multi_hop_inters_ML_1M.pt" contains multi-hop user-item interactions.
More data about "high-order interactions" can be found [here](https://drive.google.com/drive/folders/1CJdlsNuDnLiiyh4iN1eRBGRAKZ3GfxZn?usp=drive_link).
### Training
#### CF-Diff
```
cd ./CF_Diff
python main.py
```

### Inference
```
cd ./CF_Diff
python inference.py
```
## Citation  

```
@inproceedings{hou2024collaborative,
title = {Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity},
author = {Hou, Yu and Park, Jin-Duk and Shin, Won-Yong},
booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
year = {2024}
}
```
