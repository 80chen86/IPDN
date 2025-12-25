# IPDN: Image-enhanced Prompt Decoding Network for 3D Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)

NEWS:🔥IPDN is accepted at AAAI 2025 !🔥

Qi Chen, Changli Wu, Jiayi Ji, Yiwei Ma, Danni Yang, Xiaoshuai Sun

Framework:

<img src="docs\IPDN.png"/>

## Introduction
3D Referring Expression Segmentation (3D-RES) aims to
segment point cloud scenes based on a given expression.
However, existing 3D-RES approaches face two major challenges: feature ambiguity and intent ambiguity. Feature ambiguity arises from information loss or distortion during point
cloud acquisition due to limitations such as lighting and viewpoint. Intent ambiguity refers to the model’s equal treatment
of all queries during the decoding process, lacking top-down
task-specific guidance. In this paper, we introduce an Imageenhanced Prompt Decoding Network (IPDN), which leverages multi-view images and task-driven information to enhance the model’s reasoning capabilities. To address feature
ambiguity, we propose the Multi-view Semantic Embedding
(MSE) module, which injects multi-view 2D image information into the 3D scene and compensates for potential spatial
information loss. To tackle intent ambiguity, we designed a
Prompt-Aware Decoder (PAD) that guides the decoding process by deriving task-driven signals from the interaction between the expression and visual features. Comprehensive experiments demonstrate that IPDN outperforms the state-ofthe-art by 1.9 and 4.2 points in mIoU metrics on the 3D-RES
and 3D-GRES tasks, respectively.

## Installation

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n ipdn python=3.8
  conda activate ipdn
  ```

- Clone this repository

  ```
  git clone https://github.com/80chen86/IPDN.git
  ```

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl # please check the versions in the website
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install ipdn and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd ipdn/lib/
  python setup.py develop
  ```
- Compile pointnet++
  ```
  cd pointnet2
  python setup.py install --user
  cd ..
  ```
- Install Pytorch3D. You can try 'pip install pytorch3d'. If it doesn't work, you can install it from source:
  ```
  git clone git@github.com:facebookresearch/pytorch3d.git
  cd pytorch3d && pip install -e .
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` folder as follows. You need to download the ['.aggregation.json', '.sens', '.txt', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean_2.labels.ply', '_vh_clean.aggregation.json'] files. (Please note ! ! ! This will take up nearly 1TB of storage space ! ! ! If you do not have enough hard drive capacity, you can skip downloading the '.sens' files and the 2D feature extraction step, and directly download the processed 2D features.)

```
IPDN
├── data
│   ├── scannetv2
│   │   ├── scans
```

Split and preprocess point cloud data (Note! If you have not downloaded the '.sens' files, please comment out the corresponding sections in the script before running it.)

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
IPDN
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── train
│   │   ├── val
│   │   ├── processed (if you process '.sen' file)
```

Obtain image features using CLIP and project them to point. (Please modify the output of the original CLIP's visual encoder to obtain 'tokens'. See line 79 in image2point_clip.py for more imformation.)

If you haven't processed the '.sen' files or find this step too time-consuming, you can download our preprocessed features and unzip them into the 'clip-feat' folder. Click [here](https://drive.google.com/drive/folders/1WD0AlPBAxwb8zpmTeuXphRw9wrpNafUG?usp=sharing) to download.

```
cd ..
python image2point_clip.py
```

### ScanRefer dataset
Download [ScanRefer](https://github.com/daveredrum/ScanRefer) annotations following the instructions.

In the original ScanRefer annotations, all `ann_id` within each scene were individually assigned based on the corresponding `object_id`, resulting in duplicate `ann_id`. We have modified the ScanRefer annotations, and the revised annotation data, where each `ann_id` within a scene is unique, can be accessed [here](https://drive.google.com/drive/folders/1WD0AlPBAxwb8zpmTeuXphRw9wrpNafUG?usp=sharing).

Put the downloaded `ScanRefer` folder as follows.
```
IPDN
├── data
│   ├── ScanRefer
│   │   ├── ScanRefer_train_new.json
│   │   ├── ScanRefer_val_new.json
```

### Multi3DRefer dataset
Downloading the [Multi3DRefer](https://aspis.cmpt.sfu.ca/projects/multi3drefer/data/multi3drefer_train_val.zip) annotations. 

Put the downloaded `Multi3DRefer` folder as follows.
```
MDIN
├── data
│   ├── Multi3DRefer
│   │   ├── multi3drefer_train.json
│   │   ├── multi3drefer_val.json
```
There are some typos in the original text, please correct them to prevent syntax parsing errors. Or download the modified [`Multi3DRefer(New)`](https://drive.google.com/drive/folders/1WD0AlPBAxwb8zpmTeuXphRw9wrpNafUG?usp=sharing)

## Pretrained Backbone

Download [SPFormer](https://drive.google.com/drive/folders/1WD0AlPBAxwb8zpmTeuXphRw9wrpNafUG?usp=sharing) pretrained model and move it to backbones.
```
mkdir backbones
mv ${Download_PATH}/sp_unet_backbone.pth backbones/
```

## Training
For 3D-GRES:
```
bash scripts/train_gres.sh
```
For 3D-RES:
```
bash scripts/train_res.sh
```

## Inference
For 3D-GRES:
```
bash scripts/test_gres.sh
```
For 3D-RES:
```
bash scripts/test_res.sh
```

## Citation

If you find this work useful in your research, please cite:

```

```

## Models
Download pretrain models and move it to checkpoints.
|Benchmark | Task  | mIoU | Acc@0.25 | Acc@0.5 | Model |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Multi3DRes | 3D-GRES | 51.7 | 71.5 | 50.0 | [Model](https://drive.google.com/drive/folders/1WD0AlPBAxwb8zpmTeuXphRw9wrpNafUG?usp=sharing) |
| ScanRefer   | 3D-RES | 50.2 | 60.6 | 54.9 | [Model](https://drive.google.com/drive/folders/1WD0AlPBAxwb8zpmTeuXphRw9wrpNafUG?usp=sharing) |

## Ancknowledgement

Sincerely thanks for [MDIN](https://github.com/sosppxo/MDIN), [MaskClustering](https://github.com/PKU-EPIC/MaskClustering), [ReLA](https://github.com/henghuiding/ReLA), [M3DRef-CLIP](https://github.com/3dlg-hcvc/M3DRef-CLIP), [EDA](https://github.com/yanmin-wu/EDA), [SceneGraphParser](https://github.com/vacancy/SceneGraphParser), [SoftGroup](https://github.com/thangvubk/SoftGroup), [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) and [SPFormer](https://github.com/sunjiahao1999/SPFormer) repos. This repo is build upon them.
