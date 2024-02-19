## Build the conda environment

Personally, our cuda and cudnn version is 11.3, we choose python 3.8 and pytorch 1.10.0 as our environment. You can adjust your installing method based on your cuda and cudnn version.

```shell
conda create -n mvsdet python=3.8
conda activate mvsdet
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Install MMCV, MMDetection and MMSegmentation

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
```
ours：
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
```
## Install MMDetection3D based on FCAF3D

our MMDetection3D 0.15.0 is installed through building FCAF3D

```bash
git clone  https://github.com/SamsungLabs/fcaf3d.git
cd fcaf3d
pip install -r requirements/build.txt
python setup.py develop
cd {project_path}
ln -s {fcaf3d_path} mmdetection3d
```

## Install Rotated IOU

```shell
git clone https://github.com/lilanxiao/Rotated_IoU rotated_iou
cd rotated_iou
git checkout 3bdca6b20d981dffd773507e97f1b53641e98d0a
cp -r cuda_op {fcaf3d_path}/mmdet3d/ops/rotated_iou
cd {fcaf3d_path}/mmdet3d/ops/rotated_iou/cuda_op
python setup.py install 
```

## Install MinkowskiEngine v0.5.4

Our network requires the usage of MinkowskiEngine v0.5.4, following is our example, which may not be successful for all settings. You can seek for help in https://github.com/NVIDIA/MinkowskiEngine if you have trouble installing it.

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git switch -c v0.5.4
export CXX=c++; export CUDA_HOME=/usr/local/cuda-11.3; python setup.py install --blas=openblas --force_cuda
```



