# Multicam Test Summary: doll

Image dir: `/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test`

| Method | Height_mm | Cameras | Reproj_px | Volume_cm3 | Voxels | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Point Triangulation | 201.792 | Camera_Room1_2,Camera_Room1_5,Camera_Room1_7,Camera_Room1_10 | 11.946 | - | - | offset_mm=-5.0 |
| BBox Volume | 395.000 | Camera_Room1_2,Camera_Room1_5,Camera_Room1_7,Camera_Room1_10 | 11.946 | 14688.625 | 117509 | bbox_pad_px=16.9 |
| Mask Visual Hull | 265.000 | Camera_Room1_2,Camera_Room1_5,Camera_Room1_7,Camera_Room1_10 | 11.946 | 2899.625 | 23197 | mask=sam, dilate_px=12 |

## Report Paths
- Point Triangulation: `/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/doll_height_outputs/multicam_doll_height_report.json`
- BBox Volume: `/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/doll_bbox_volume_outputs/multicam_doll_bbox_volume_report.json`
- Mask Visual Hull: `/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/doll_mask_visual_hull_outputs/multicam_doll_mask_visual_hull_report.json`
