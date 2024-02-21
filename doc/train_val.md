# Training and Validation procedures of CN-RMA

The training procedure is complicated, if you just want to get the results for comparison, or want to reproduce our result, I recommend you directly download the results, or directly use our pre-trained weights to validate.(TODO)

Before training and validating our method, please ensure you have successfully built up the environment and processed the data.

## Step 1: Pretraining the Atlas reconstruction network

If you want to skip step 1, you can directly download our pre-trained weights on ScanNet and ARKitScenes.(TODO)

First, you should download the resnet weights: R-50.pth, and change the `R50_path` variable of the `projects/configs/mvsdetection/atlas_recon_scannet.py` and the `projects/configs/mvsdetection/atlas_recon_arkit.py`.

Second, you should construct a work directory to store your results and checkpoints, and change the `work_dir` variable of the config files.

Finally, you can run the pretraining procedure

ScanNet:

```shell
bash dist_train.sh ./projects/configs/mvsdetection/atlas_recon_scannet.py 4
```

ARKitScenes:

```shell
bash dist_train.sh ./projects/configs/mvsdetection/atlas_recon_arkit.py 4
```

## Step 2: Pretraining the FCAF3D detection network

If you want to skip Steps 1 and 2, you can directly download our pre-trained base weights on ScanNet and ARKitScenes.(TODO)

For convenience, we **dump the point clouds with features** based on our pre-trained reconstruction network and our ray marching method, and **directly use the FCAF3D code** to pre-train the FCAF3D network. 

### 2.1 Dump the point clouds with features

First, you need to convert your pre-trained reconstruction weight to a universal version. Before this, you should `download` the checkpoint template (PATH)(TODO).

```shell
python ./data_prepare/combine_models.py --atlas_model_path {your_pretrained_recon_ckpt_path} --fcaf3d_model_path none --full_model_path {template_path} --result_path {your_converted_weight_path} --result_type full
```

Then, you can dump the point cloud with features with the converted checkpoint. The config files are `./projects/configs/mvsdetection/scannet_middle.py` (ScanNet) and `./projects/configs/mvsdetection/arkit_middle.py` (ARKitScenes)

Same as previous steps, you should change the `R50_path`, `work_dir` variables to your desire. Additionally, you should change the `MIDDLE_SAVE_PATH` to the path that you want to save your point cloud with features. If you want to visualize the points, you can also change the `MIDDLE_VISUALIZE_PATH`.

ScanNet:

```shell
bash dist_test.sh ./projects/configs/mvsdetection/scannet_middle.py {your_converted_weight_path} 4 
```

ARKitScenes:

```shell
bash dist_test.sh ./projects/configs/mvsdetection/arkit_middle.py {your_converted_weight_path} 4 
```

After this, you will dump all the training data into the path. You can dump the validating data by changing the `data("test")("ann_file")` variable to `xxxx_val.pkl`.



The point cloud with features has the name xxxxxxxx_vert.npy. After dumping the point clouds with features, you should also copy the bounding boxes and semantic, instance annotations of your ARKitScenes and ScanNet dataset to the path, in order to train the FCAF3D network. The path that saves the point cloud with features should have the following structure:

```
├── {your_save_path}
│   ├──xxxxxxxx_aligned_bbox.npy
│   ├──xxxxxxxx_axis_align_matrix.npy
│   ├──xxxxxxxx_axis_ins_label.npy
│   ├──xxxxxxxx_axis_sem_label.npy
│   ├──xxxxxxxx_axis_unaligned_bbox.npy
│   ├──xxxxxxxx_axis_vert.npy(this is your point cloud with feature, the others are the ground truth annotations)
```

### 2.2 Pretrain the FCAF3D network

You should build the environment and process the data according to [FCAF3D](https://github.com/SamsungLabs/fcaf3d) first. The only difference is that you should change the `scannet_instance_data` and the `arkit_instance_data` files, which denotes the ground truth bounding box and segmentation annotations, to the path that you save your point cloud with features in Step 2.1.

You should put the `./fcaf3d/fcaf3d_middle_arkit.py` and `./fcaf3d/fcaf3d_middle_scannet.py` of our code repository to the `./configs/fcaf3d` directory of the FCAF3D repository, and put the `./fcaf3d/scannet_dataset.py` and `./fcaf3d/arkit_dataset.py` to the `./mmdet3d/datasets` directory of the FCAF3D repository.

Then, you can run the FCAF3D training code:

ScanNet:

```shell
bash tools/dist_train.sh configs/fcaf3d/fcaf3d_middle_scannet.py 4
```

ARKitScenes

```shell
bash tools/dist_train.sh configs/fcaf3d/fcaf3d_middle_arkit.py 4
```

### 2.3 Combine the pre-trained Atlas reconstruction network and the FCAF3D detection network

After pretraining the FCAF3D detection network, you should combine the Atlas reconstruction network and the FCAF3D detection network together, as the basis before final fine-tuning.

```shell
python ./data_prepare/combine_models.py --atlas_model_path {your_pretrained_recon_ckpt_path} --fcaf3d_model_path {your_pretrained_detection_ckpt_path} --full_model_path {template_path} --result_path {your_converted_weight_path} --result_type full
```

## Step 3: Finetuning the full network

After Steps 1 and 2, you should have successfully constructed the basis of the step. If you want to skip Steps 1 and 2, you can directly download our pre-trained base weights on ScanNet and ARKitScenes.(TODO)

The config files are `./projects/configs/mvsdetection/ray_marching_scannet.py` (ScanNet) and `./projects/configs/mvsdetection/ray_marching_arkit.py` (ARKitScenes). Same as previous steps, you should change the `R50_path`, `work_dir` variables to your desire. Additionally, you should set the `load_from`  variable to the pre-trained base weight path. 

ScanNet:

```shell
bash dist_train.sh ./projects/configs/mvsdetection/ray_marching_scannet.py 4
```

ARKitScenes:

```shell
bash dist_train.sh ./projects/configs/mvsdetection/ray_marching_arkit.py 4
```

## Step 4: Evaluating the results

To Skip Steps 1, 2, and 3, you can directly download our final checkpoints. (TODO) And you can directly download our final results. (TODO)

First, you should get the result bounding boxes (without nms post-processing).

ScanNet:

```shell
bash dist_test.sh projects/configs/mvsdetection/ray_marching_scannet.py {scannet_best.pth} 4
```

ARKitScenes:

```shell
bash dist_test.sh projects/configs/mvsdetection/ray_marching_arkit.py {arkit_best.pth} 4
```

After this, you should do nms post-processing to the data by running:

```shell
python ./post_process/nms_bbox.py --result_path {your_work_dir}/results
```

The pc_det_nms does not always work very well, if it fails, just run it again and again.... 

You can then evaluate the results by running

```shell
./post_process/evaluate_bbox.py --dataset {arkit/scannet} --data_path {your_arkit_or_scannet_source_path} --result_path {your_work_dir}/results
```

You can visualize the results by running

```shell
./post_process/visualize_results.py --dataset {arkit/scannet} --data_path {your_arkit_or_scannet_source_path} --save_path {your_work_dir}/results
```

if the nms fails, you can discover many bounding boxes very close to each other on the visualized results, then you can run the nms again.