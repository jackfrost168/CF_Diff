# CF-Diff
This is the pytorch implementation of our paper at SIGIR 2024:
> [Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity](https://arxiv.org/pdf/2404.14240)
> 
> Hou, Yu and Park, Jin-Duk and Shin, Won-Yong

The implementation of diffusion model part is referred to [DiffRec](https://github.com/YiyanXu/DiffRec/tree/main).
## Environment
- Anaconda 3
- python 3.8.17
- pytorch 1.13.1
- numpy 1.24.3
- math

## Usage
### Data
The user-item interaction data is in './datasets' folder
The file of sec_hop_inters_ML_1M.pt contains the information of second-hop user-item interactions

### Training
#### CF-Diff
```
cd ./CF-Diff
python main.py
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
