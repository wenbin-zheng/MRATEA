# SCASMF
## Spatial Coordinate Attention and Sparse Mask Fusion for Incomplete Multimodality Brain Tumor Segmentation
Paper: Spatial Coordinate Attention and Sparse Mask Fusion for Incomplete Multimodality Brain Tumor Segmentation





### Requirements
All our experiments are implemented based on the PyTorch framework with one 24G NVIDIA Geforce RTX 3090 GPUs, and we recommend installing the following package versions:
- python=3.10
- pytorch=1.12.1
- torchvision=0.13.1

### dataset
if you want to download the BraTS2018  dataset, you can access the link:https://aistudio.baidu.com/aistudio/datasetdetail/64660

if you want to download the BraTS2018  dataset, you can access the link:https://aistudio.baidu.com/datasetdetail/225057

- To obtain the 'BraTS20xx_Training_none_npy' folders, please run the 'preprocess.py' script.


## Training

**SCATrans**

- Changing the paths and hyperparameters in  ``train.py`` and ``predict.py``.
- Set different splits for BraTS20xx in ``train.py``.
- Then run:

```bash
  python train.py --datapath='BRATS2018_Training_none_npy',--dataname='MICCAI_BraTS_2018_Data_Training' -modilities="SCATrans" --epochs=1000 --learing rate=0.0002 --batchsize=2
```
  
## Evaluation

Checking the relevant paths in path in  ``eval.py``.
