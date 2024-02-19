### Prepare ARKitScenes Data for CN-RMA

We follow the procedure of ScanNet.

1. Download ARKitScenes data [HERE](https://github.com/apple/ARKitScenes). Only 3dod data is required. 

   ```
   ARKit/3dod/Training(Validation)
   ├── xxxxxxxx
   │   ├──xxxxxxxx_frames
   │   │   ├──lowres_depth
   │   │   │  ├──xxxxxxxx_xxxx.xxx.png
   │   │   ├──lowres_wide
   │   │   │  ├──xxxxxxxx_xxxx.xxx.png
   │   │   ├──lowres_wide_intrinsics
   │   │   │  ├──xxxxxxxx_xxxx.xxx.pincam
   │   │   ├──lowres_wide.traj
   │   ├──xxxxxxxx_3dod_annotation.json
   │   ├──xxxxxxxx_3dod_mesh.ply
   ```

   Link or move the 'Training' and 'Validation' folder to `./data/arkit`

2. Extract point clouds and annotations.

   ```shell
   cd {project_path}/data_prepare/arkit
   python load_arkit_data.py --output_folder {your_extract_path} --data_path {your_arkit_path}
   ln {your_extract_path} {project_path}/data/arkit/arkit_instance_data
   ```

3. Generate TSDF data

   ```shell
   cd {project_path}/data_prepare/arkit
   python generate_tsdf.py --data_path {your_arkit_path} --save_path {your_tsdf_save_path}
   ln -s {your_tsdf_save_path} {project_path}/data_prepare/arkit/atlas_tsdf
   ```

4. Aggregate all data into pkl files for mmdetection3d usage

```bash
cd {project_path}/data_prepare/arkit
python aggregate_data.py --data_path {project_path}/data/arkit --save_path {project_path}/data/arkit
```

The directory structure after data-processing should be as below

```
arkit
├── Training
├── Validation
├── arkit_instance_data
│   ├──xxxxxxxx_aligned_bbox.npy
│   ├──xxxxxxxx_axis_align_matrix.npy
│   ├──xxxxxxxx_axis_ins_label.npy
│   ├──xxxxxxxx_axis_sem_label.npy
│   ├──xxxxxxxx_axis_unaligned_bbox.npy
│   ├──xxxxxxxx_axis_vert.npy
├── atlas_tsdf
│   ├── xxxxxxxx
│   │   ├──info.json
│   │   ├──tsdf_04.npz
│   │   ├──tsdf_08.npz
│   │   ├──tsdf_16.npz
├── arkit_infos_train.pkl
├── arkit_infos_val.pkl
```