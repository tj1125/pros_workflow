# Triplet Height Ranking (doll, apple, wine)

Image dir: `/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1`
Usable cameras: `11` / calibrated `12`
Missing image cameras: `Camera_Room1_12`

| Rank | Camera Combo | Valid | Mean Abs Err (mm) | Max Abs Err (mm) | Mean Reproj (px) | Doll Est/Err | Apple Est/Err | Wine Est/Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Camera_Room1_5,Camera_Room1_7,Camera_Room1_8 | 3 | 109.125 | 199.383 | 9.816 | 194.685 / 0.315 | 139.676 / 127.676 | 223.383 / 199.383 |
| 2 | Camera_Room1_5,Camera_Room1_8,Camera_Room1_9 | 3 | 111.664 | 202.762 | 10.986 | 197.470 / 2.470 | 141.760 / 129.760 | 226.762 / 202.762 |
| 3 | Camera_Room1_3,Camera_Room1_8,Camera_Room1_10 | 3 | 112.632 | 206.778 | 5.231 | 194.336 / 0.664 | 142.453 / 130.453 | 230.778 / 206.778 |
| 4 | Camera_Room1_2,Camera_Room1_3,Camera_Room1_9 | 3 | 112.886 | 207.796 | 7.857 | 193.442 / 1.558 | 141.304 / 129.304 | 231.796 / 207.796 |
| 5 | Camera_Room1_3,Camera_Room1_9,Camera_Room1_10 | 3 | 113.247 | 205.695 | 6.502 | 198.417 / 3.417 | 142.628 / 130.628 | 229.695 / 205.695 |
| 6 | Camera_Room1_5,Camera_Room1_8,Camera_Room1_10 | 3 | 113.504 | 205.498 | 13.209 | 199.882 / 4.882 | 142.131 / 130.131 | 229.498 / 205.498 |
| 7 | Camera_Room1_3,Camera_Room1_8,Camera_Room1_9 | 3 | 113.538 | 206.573 | 6.736 | 193.901 / 1.099 | 144.941 / 132.941 | 230.573 / 206.573 |
| 8 | Camera_Room1_3,Camera_Room1_7,Camera_Room1_8 | 3 | 113.567 | 207.577 | 4.146 | 193.959 / 1.041 | 144.081 / 132.081 | 231.577 / 207.577 |
| 9 | Camera_Room1_5,Camera_Room1_9,Camera_Room1_10 | 3 | 113.798 | 204.524 | 10.587 | 203.346 / 8.346 | 140.523 / 128.523 | 228.524 / 204.524 |
| 10 | Camera_Room1_2,Camera_Room1_3,Camera_Room1_7 | 3 | 114.001 | 215.242 | 6.672 | 193.739 / 1.261 | 137.500 / 125.500 | 239.242 / 215.242 |
| 11 | Camera_Room1_3,Camera_Room1_5,Camera_Room1_8 | 3 | 114.170 | 212.307 | 11.218 | 196.854 / 1.854 | 140.350 / 128.350 | 236.307 / 212.307 |
| 12 | Camera_Room1_2,Camera_Room1_3,Camera_Room1_10 | 3 | 114.285 | 214.819 | 6.368 | 193.709 / 1.291 | 138.743 / 126.743 | 238.819 / 214.819 |
| 13 | Camera_Room1_3,Camera_Room1_7,Camera_Room1_10 | 3 | 114.430 | 208.016 | 4.461 | 199.984 / 4.984 | 142.290 / 130.290 | 232.016 / 208.016 |
| 14 | Camera_Room1_8,Camera_Room1_9,Camera_Room1_10 | 3 | 114.649 | 205.579 | 3.792 | 197.850 / 2.850 | 147.519 / 135.519 | 229.579 / 205.579 |
| 15 | Camera_Room1_1,Camera_Room1_3,Camera_Room1_9 | 3 | 114.713 | 212.724 | 7.281 | 198.540 / 3.540 | 139.874 / 127.874 | 236.724 / 212.724 |
| 16 | Camera_Room1_1,Camera_Room1_3,Camera_Room1_8 | 3 | 114.776 | 215.434 | 6.473 | 194.434 / 0.566 | 140.327 / 128.327 | 239.434 / 215.434 |
| 17 | Camera_Room1_3,Camera_Room1_7,Camera_Room1_9 | 3 | 114.846 | 208.002 | 7.039 | 199.816 / 4.816 | 143.720 / 131.720 | 232.002 / 208.002 |
| 18 | Camera_Room1_5,Camera_Room1_7,Camera_Room1_9 | 3 | 115.005 | 205.363 | 14.265 | 203.989 / 8.989 | 142.662 / 130.662 | 229.363 / 205.363 |
| 19 | Camera_Room1_3,Camera_Room1_5,Camera_Room1_7 | 3 | 115.400 | 211.156 | 12.303 | 202.364 / 7.364 | 139.680 / 127.680 | 235.156 / 211.156 |
| 20 | Camera_Room1_2,Camera_Room1_5,Camera_Room1_8 | 3 | 115.685 | 215.414 | 12.684 | 196.942 / 1.942 | 141.700 / 129.700 | 239.414 / 215.414 |

## Best Triplet By Target
- `doll`: `Camera_Room1_5,Camera_Room1_7,Camera_Room1_8` estimated `194.685 mm`, abs error `0.315 mm`
- `apple`: `Camera_Room1_3,Camera_Room1_4,Camera_Room1_6` estimated `131.492 mm`, abs error `119.492 mm`
- `wine`: `Camera_Room1_5,Camera_Room1_7,Camera_Room1_8` estimated `223.383 mm`, abs error `199.383 mm`
