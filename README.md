# D-PETA
SFDAKD(Source Free Domain Adaptation Keypoint Detection)

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
**we use single 3090 gpu to complete the experiment**

## 3. Data
**[SURREAL(source data)](https://drive.google.com/file/d/1mwZGHMolb9utu60pUjLSPszfRfK1n1XX/view?usp=drive_link)**

`@INPROCEEDINGS{varol17_surreal,
  title = {Learning from Synthetic Humans},
  author = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year = {2017}
}`


**[LSP(target data)](https://drive.google.com/file/d/1oXHX57tH7q-8F08sQbm8fwXJEcUWjtF9/view?usp=drive_link)**

`@inproceedings{Johnson10,
   title = {Clustered Pose and Nonlinear Appearance Models for Human Pose Estimation},
   author = {Johnson, Sam and Everingham, Mark},
   year = {2010},
   booktitle = {Proceedings of the British Machine Vision Conference},
   note = {doi:10.5244/C.24.12}
}`

### usage of data

**You should use the source data to train source model by human_src.py and use the target data to complete the adaptation by human_tgt.py.**

## 4. Result
![image](https://github.com/user-attachments/assets/42d25d41-b631-4788-9883-2a48ab2816dc)

## 5.Acknowledgement
This repo is modified from open source SFDAKD codebase [MAPS](https://github.com/YuheD/MAPS?tab=readme-ov-file), [SFDAHPE](https://github.com/davidpengucf/SFDAHPE.).

