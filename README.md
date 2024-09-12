
# **OccFaçade: enabling precise building façade parsing in large urban scenes with occlusion**

paper: https://www.tandfonline.com/doi/abs/10.1080/01431161.2024.2391589

We propose a building facade parsing method called OccFacade and a dataset MeshFacade that maps Mesh to images.
![fig9-2](https://github.com/yueyisui/OccFacade/assets/64672040/dac3b463-7656-4870-b15d-2403024fd409)

First you need to run the following to get the code.

```
git clone git@github.com:yueyisui/OccFacade.git
cd OccFacade
```



## Data Preparation

Download the [dataset](https://drive.google.com/drive/folders/13VPR8n3uJAdjUy-3rX3LG0ueLFg4_Iy4?usp=drive_link).

The dataset includes our divided MeshFacade dataset and other open source datasets.

Then you need to put the downloaded data into the **data** folder.



## Train and Test

First check the file contents under the configs folder, then run the code by typing the following in the terminal.

```
python run_main.py
```

Most parameters are in configs and can be modified as needed

If the parameter **pred** is not commented, it is test mode, otherwise it is training mode.

If you need to test, you need to download the **weight** file and place it under the logs file.

Download the [weight](https://drive.google.com/drive/folders/12mh7ksj-1WJjMALXyIlrApIPee_sMgCu?usp=drive_link).



## Publication

```
@article{doi:10.1080/01431161.2024.2391589,
author = {Yongjun Zhang, Dongdong Yue, Xinyi Liu, Siyuan Zou, Weiwei Fan and Zihang Liu},
title = {OccFaçade: enabling precise building façade parsing in large urban scenes with occlusion},
journal = {International Journal of Remote Sensing},
volume = {45},
number = {18},
pages = {6651--6674},
year = {2024},
publisher = {Taylor \& Francis},
doi = {10.1080/01431161.2024.2391589}
```



