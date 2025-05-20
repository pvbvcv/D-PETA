# D-PETA
SFDAHPE(D-PETA)

## 1. Prerequisites

### Dependencies
- dinov2==0.0.1.dev0
- loralib==0.1.2
- opencv-python==4.9.0.80
- scikit-image==0.17.2
- scikit-learn==1.3.2
- scipy==1.10.1
- torch==2.0.1
- torchaudio==2.1.1+cu118
- torchvision==0.16.1+cu118
- triton==2.0.0
- typing-extensions==4.8.0
- typing-inspect==0.9.0
- urllib3==1.26.13
- webcolors==1.13

**You can use the [environment.yml](https://github.com/pvbvcv/D-PETA/blob/main/environment.yml) to create your conda environment.**

## 2. Training
### source model training

#### single GPU  `python human_src.py`
#### multi GPU `CUDA_VISIBLE_DEVICES=x,x,x,x python human_src.py`

### adaptation

#### single GPU  `python human_tgt.py`
#### multi GPU `CUDA_VISIBLE_DEVICES=x,x,x,x python human_tgt.py`
