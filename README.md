# CN-RMA: Combined Network with Ray Marching Aggregation for 3D Indoor Object Detection from Multi-view Images  
This repository is an official implementation of CN-RMA

## Results 

| dataset | mAP@0.25 | mAP@0.5 |                            config                            |
| :-----: | :------: | :-----: | :----------------------------------------------------------: |
| ScanNet |   58.6   |  36.8   | [config](./projects/configs/mvsdetection/ray_marching_scannet.py) |
|  ARKit  |   67.6   |  56.5   | [config](./projects/configs/mvsdetection/ray_marching_arkit.py) |

Configuration, data processing and running the entire project is complicated. We provide all the detection results, visualization results and checkpoints of the validation set of the two datasets at Baidu Netdisk and Google Drive.(TODO)

## Prepare

* Environments  

  Linux, Python==8, CUDA == 11.3, pytorch == 1.10.0, mmdet3d == 0.15.0, MinkowskiEngine == 0.5.4   

  This implementation is built based on the [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework and can be constructed as the [install.md](./doc/install.md).

* Data   
Follow the mmdet3d to process the ScanNet and ARKitScenes datasets. You can process those datasets following [scannet.md](./doc/scannet.md) and [arkit.md](./doc/arkit.md).


* Pretrained weights   
The required pretrained weights are put at here(TODO).

* After preparation, you will be able to see the following directory structure:  
  ```
  CN-RMA
  ├── mmdetection3d
  ├── projects
  │   ├── configs
  │   ├── mvsdetection
  ├── tools
  ├── data
  │   ├── scannet
  │   ├── arkit
  ├── doc
  │   ├── install.md
  │   ├── arkit.md
  │   ├── scannet.md
  │   ├── train_val.md
  ├── README.md
  ├── data_prepare
  ├── post_process
  ├── dist_test.sh
  ├── dist_train.sh
  ├── test.py
  ├── train.py
  ```

##  How to Run

To evaluate our method on ScanNet, you can download the pre-trained checkpoint, set the 'work_dir' of `projects/configs/mvsdetection/ray_marching_scannet.py` to your desired path, and run:

```shell
bash dist_test.sh projects/configs/mvsdetection/ray_marching_scannet.py {scannet_best.pth} 4
```

Similarly, to evaluate on ARKitScenes, you should download the checkpoint, set the 'work_dir' of `projects/configs/mvsdetection/ray_marching_arkit.py` to your desired path, and run:

```shell
bash dist_test.sh projects/configs/mvsdetection/ray_marching_arkit.py {arkit_best.pth} 4
```

After this, you should do nms post-processing to the data by running:

```shell
python ./post_process/nms_bbox.py --result_path {your_work_dir}/results
```

The pc_det_nms do not always work very well, if it fails, just run it again and again.... 

You can then evaluate the results by running

```shell
./post_process/evaluate_bbox.py --dataset {arkit/scannet} --data_path {your_arkit_or_scannet_source_path} --result_path {your_work_dir}/results
```

And you can visualize the results by running

```shell
./post_process/visualize_results.py --dataset {arkit/scannet} --data_path {your_arkit_or_scannet_source_path} --save_path {your_work_dir}/results
```

if the nms fails, you can discover many bounding boxes very close to each other on the visualized results, then you can run the nms again.



Training the network from scratch is complicated. If you want to train the network from scratch, please follow [train_val.md](./doc/train_val.md) 

## Citation

If you find this project useful for your research, please consider citing: 
```bibtex   
//TODO
```
## Contact
If you have any questions, feel free to open an issue or contact us at shenguanlin1999@163.com
