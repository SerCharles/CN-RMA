
### Prepare ScanNet Data for CN-RMA

We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to `./data/scannet.` If you are performing segmentation tasks and want to upload the results to its official [benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/), please also link or move the 'scans_test' folder to `./data/scannet`.

2. Extract point clouds and annotations.

   ```shell
   cd {project_path}/data_prepare/scannet
   python batch_load_scannet_data.py --output_folder {your_extract_path} --train_scannet_dir {project_path}/data/scannet/scans --test_scannet_dir {project_path}/data/scannet/scans_test --label_map_file {project_path}/data/scannet/meta_data/scannetv2-labels.combined.tsv --train_scan_names_file {project_path}/data/scannet/meta_data/scannet_train.txt --test_scan_names_file {project_path}/data/scannet/meta_data/scannetv2_test.txt
   ln -s {your_extract_path} {project_path}/data/scannet/scannet_instance_data
   ```

   Add the `--max_num_point 50000` flag if you only use the ScanNet data for the detection task. It will downsample the scenes to less points.

3. Extract RGB image with poses

   ```shell
   cd {project_path}/data_prepare/scannet
   python extract_posed_images.py --data_root {project_path}/data/scannet --save_path {your_posed_image_path}
   ln -s {your_posed_image_path} {project_path}/data/scannet/posed_images
   ```

   Add `--max-images-per-scene -1` to disable limiting number of images per scene. ScanNet scenes contain up to 5000+ frames per each. After extraction, all the .jpg images require 2 Tb disk space. The recommended 300 images per scene require less then 100 Gb. For example multi-view 3d detector ImVoxelNet samples 50 and 100 images per training and test scene.

4. Generate TSDF data

   ```shell
   cd {project_path}/data_prepare/scannet
   python generate_tsdf.py --data_path {project_path}/data/scannet --save_path {your_tsdf_save_path}
   ln -s {your_tsdf_save_path} {project_path}/data_prepare/scannet/atlas_tsdf
   ```

4. Aggregate all data into pkl files for mmdetection3d usage

```bash
cd {project_path}/data_prepare/scannet
python aggregate_data.py --data_path {project_path}/data/scannet --save_path {project_path}/data/scannet
```

The directory structure after data-processing should be as below

```
scannet
├── meta_data
├── scans
├── scans_test
├── scannet_instance_data
│   ├──scenexxxx_xx_aligned_bbox.npy
│   ├──scenexxxx_xx_axis_align_matrix.npy
│   ├──scenexxxx_xx_axis_ins_label.npy
│   ├──scenexxxx_xx_axis_sem_label.npy
│   ├──scenexxxx_xx_axis_unaligned_bbox.npy
│   ├──scenexxxx_xx_axis_vert.npy
├── posed_images
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.txt(pose)
│   │   ├── xxxxxx.jpg(image)
│   │   ├── xxxxxx.png(depth)
│   │   ├── intrinsic.txt
├── atlas_tsdf
│   ├── scenexxxx_xx
│   │   ├──info.json
│   │   ├──tsdf_04.npz
│   │   ├──tsdf_08.npz
│   │   ├──tsdf_16.npz
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
├── scannet_infos_test.pkl
```
