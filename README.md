<<<<<<< HEAD
# IMoS: Intent-driven fullbody motion synthesis for Human-Object interaction (Eurographics 2023).

[Paper](https://arxiv.org/pdf/2212.07555.pdf) | 
[Video](https://www.youtube.com/watch?v=3Ngi9k41-7c) | 
[Project Page](https://vcai.mpi-inf.mpg.de/projects/IMoS/)

<img src="teaser.jpg" alt="teaser image" />

Our code is tested on the following OS: 
* Ubuntu 20.04 LTS
* Windows 10, 11

## Pre-requisites
### Code
* Python >= 3.8
* Pytorch >= 1.11
* conda >= 4.9.2 (optional but recommended)

## Getting started

Follow the commands to create conda environment:
```
conda create -n IDMS python=3.8
conda activate IDMS
conda install -c pytorch pytorch=1.11 torchvision cudatoolkit=11.3
pip install -r requirements.txt
```
For pytorch3D installation refer to https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
For Ubuntu OpenGL is to be installed as:
```apt-get update
apt-get install python3-opengl
```
1. Follow the instructions on the [SMPL-X](https://smpl-x.is.tue.mpg.de) website to download SMPL-X model and keep the downloaded files under the 'smplx_model' folder.

2. Download the GRAB dataset from the [GRAB website](https://grab.is.tue.mpg.de), and follow the instructions there to extract the files.
Save the raw datas in '../DATASETS/GRAB'. 

3. To pre-process the GRAB dataset for our setting run: 
```python src/data_loader/dataset_preprocess.py ```


4. To evaluate our pre-trained model, run:
```    python src/evaluate/eval.py ``` 
    
5. a. To train the 'Arm Synthesiss module', run:
```    python src/train/train_arms.py ```
	b. To train the 'Body Synthesis module', run:
```    python src/train/train_body.py ```

Keep all parameters to it's default values. You can find them in src/tools/argUtils.py. 

6. To generate the '.npy' files with the synthesized motions run:
```    python src/test/test_synthesis.py ```
The pretrained model path is hardcoded in the 'args.load' parameter in the code already. Change the parameter for newly trained models.

5. To visualize some results of our paper, run:
```    python src/visualize/render_smplx.py```


=======
# IMoS
Intent-Driven Fullbody Motion Synthesis for Human-Object Interaction

Coming soon!
>>>>>>> e8954420519f1698e7556e63b0c7c2ff03df8494
