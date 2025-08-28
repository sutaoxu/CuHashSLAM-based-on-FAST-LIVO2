CuHashSLAM (based on FAST-LIVO2)
--------------------------------

This repository contains a GPU-accelerated prototype implementation
of voxel-based LiDAR–IMU SLAM components, derived from FAST-LIVO2:

  FAST-LIVO2: https://github.com/hku-mars/FAST-LIVO2

Acknowledgements:
  - FAST-LIVO2 authors for the original LiDAR/IMU/vision odometry framework.
  - NVIDIA cuCollections: https://github.com/NVIDIA/cuCollections

Primary modifications by: <Sutao Xu @ HIT>
  - include/voxel_map_cuda.h        (new)
  - src/voxel_map_cuda.cu           (new)
  - modified: src/LIVMapper.cpp     (small additions)

License:
  This project is distributed under the GNU General Public License v2 (GPLv2).

  See LICENSE for full license text.
