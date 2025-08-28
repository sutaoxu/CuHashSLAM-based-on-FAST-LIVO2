/*
This file is part of CuHashSLAM
*/

#ifndef VOXEL_MAP_CUDA_H
#define VOXEL_MAP_CUDA_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>
#include <utility>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math_constants.h>
#include <unordered_map>
#include "common_lib.h"

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdint>


#define VOXELMAP_CUDA_HASH_P 116101
#define VOXELMAP_CUDA_MAX_N 10000000000
#define TOTAL_CAPACITY 20000

__constant__ double d_parameter1[27];
__constant__ double d_parameter2[24];

__device__ int voxel_plane_cuda_id = 0;
__device__ int root_voxel_index = 0;
__device__ int leaf_voxel_index = 0;

static int voxelx, voxely, voxelz;

enum ParaIndex
{
    ROT_END = 0,
    ROT_VAR = 9,
    T_VAR = 18,
    EXTR = 0,
    EXTT = 9,
    PROPAGAT_ROT_END = 12,
    PROPAGAT_POS_END = 21
};

enum PvIndex
{
    BODY_COV_LIST = 0,
    CROSS_MAT_LIST = 9,
};

enum PtplIndex
{
    POINT_B = 0,
    CENTER = 3,
    NORMAL = 6,
    BODY_COV = 9,
    PLANE_VAR = 18,
    DIST_TO_PLANE = 54
};

enum PwvIndex
{
    PWV_POINT_B = 0,
    PWV_POINT_I = 3,
    PWV_POINT_W = 6,
    PWV_VAR_NOSTATE = 9,
    PWV_BODY_VAR = 18,
    PWV_VAR = 27,
    PWV_POINT_CROSSMAT = 36,
    PWV_NORMAL = 45,
    PWV_TOTAL_NUM = 48
};

struct Mat33
{
    double mat[9];
};

struct ParametersCuda
{
    double extR[9];
    double extT[3];
};

__constant__ ParametersCuda d_ParametersCuda;

struct StatesGroupCuda
{
    double rot_end[9];
    double pos_end[3];
    double vel_end[3];
    double bias_g[3];
    double bias_a[3];
    double gravity[3];
    double inv_expo_time;
    double cov[19*19];

};

__constant__ StatesGroupCuda d_state;
__constant__ StatesGroupCuda d_state_propagate;

struct PointCloudXYZICuda
{
    float x, y, z;
    float normal_x, normal_y, normal_z;
    float curvature;
    float intensity;
};

typedef struct pointWithVarCuda
{
    double point_b[3];
    double point_i[3];
    double point_w[3];
    double var_nostate[9];
    double body_var[9];
    double var[9];
    double point_crossmat[9];
    double normal[3];
} pointWithVarCuda; 

typedef struct VoxelMapConfigCuda
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfigCuda;

typedef struct PointToPlaneCuda
{
    double point_b_[3];
    double point_w_[3];
    double normal_[3];
    double center_[3];
    double plane_var_[36];
    double body_cov_[9];
    double d_;
    double eigen_value_;
    int layer_;
    float dis_to_plane_;
    bool is_valid_;
} PointToPlaneCuda;

typedef struct VoxelPlaneCuda
{
    double center_[3];
    double normal_[3];
    double y_normal_[3];
    double x_normal_[3];
    double covariance_[9];
    double plane_var_[36];
    float radius_;
    float min_eigen_value_;
    float mid_eigen_value_;
    float max_eigen_value_;
    float d_;
    int points_size_;
    int id_;
    bool is_plane_;
    bool is_init_;
    bool is_update_;
} VoxelPlaneCuda;

struct VOXEL_LOCATION_CUDA
{
    int64_t x, y, z;
};

struct VOXEL_LOCATION_CUDA_EQUAL
{
    __host__ __device__ bool operator()(VOXEL_LOCATION_CUDA const *a, VOXEL_LOCATION_CUDA const *b) const noexcept
    {
        return a->x==b->x && a->y==b->y && a->z==b->z;
    }
};

struct VOXEL_LOCATION_CUDA_HASH
{
    __host__ __device__ int64_t operator()(VOXEL_LOCATION_CUDA const *a) const noexcept
    {
        return (((a->z * VOXELMAP_CUDA_HASH_P) % VOXELMAP_CUDA_MAX_N + a->y) * VOXELMAP_CUDA_HASH_P) % VOXELMAP_CUDA_MAX_N + a->x;
    }
};

struct RootVoxelCuda
{
    pointWithVarCuda temp_points_[100];
    double voxel_center_[3];
    VOXEL_LOCATION_CUDA leaf_voxel_[8];
    VoxelPlaneCuda plane_;
    int layer_init_num_[5];
    int points_size_threshold_;
    int update_size_threshold_;
    int new_points_;
    float quater_length_;
    float planer_threshold_;
    int update_enable_;
    int leaf_enable_;
    int init_octo_;
    int is_valid_;
};

struct LeafVoxelCuda
{
    pointWithVarCuda temp_points_[50];
    double voxel_center_[3];
    VOXEL_LOCATION_CUDA parent_voxel_;
    VoxelPlaneCuda plane_;
    int layer_init_num_[5];
    int points_size_threshold_;
    int update_size_threshold_;
    int new_points_;
    float quater_length_;
    float planer_threshold_;
    int update_enable_;
    int init_octo_;
    int is_valid_;
};

constexpr VOXEL_LOCATION_CUDA* ROOT_VOXEL_EMPTY_KEY = nullptr;
constexpr RootVoxelCuda* ROOT_VOXEL_EMPTY_VALUE = nullptr;
constexpr VOXEL_LOCATION_CUDA* LEAF_VOXEL_EMPTY_KEY = nullptr;
constexpr LeafVoxelCuda* LEAF_VOXEL_EMPTY_VALUE = nullptr;
// static VOXEL_LOCATION_CUDA __erased_sentinel;
// constexpr VOXEL_LOCATION_CUDA* ERASED_KEY = &__erased_sentinel;

class VoxelMapManagerCuda
{
    public:
    VoxelMapManagerCuda() = default;
    VoxelMapConfigCuda config_setting_cuda_;
    int current_frame_id_ = 0;
    ros::Publisher voxel_map_pub_;

    VOXEL_LOCATION_CUDA* d_root_voxel_location_cuda = nullptr;      
    RootVoxelCuda* d_root_voxel_cuda = nullptr;                     
    VOXEL_LOCATION_CUDA* d_leaf_voxel_location_cuda = nullptr;      
    LeafVoxelCuda* d_leaf_voxel_cuda = nullptr;

    VOXEL_LOCATION_CUDA* h_root_voxel_location_cuda = nullptr;
    RootVoxelCuda* h_root_voxel_cuda = nullptr;
    VOXEL_LOCATION_CUDA* h_leaf_voxel_location_cuda = nullptr;
    LeafVoxelCuda* h_leaf_voxel_cuda = nullptr;

    PointCloudXYZICuda* h_feats_undistort_ = nullptr;
    PointCloudXYZICuda* d_feats_undistort_ = nullptr;        
    PointCloudXYZICuda* h_feats_down_body_ = nullptr;
    PointCloudXYZICuda* d_feats_down_body_ = nullptr;        
    PointCloudXYZICuda* h_feats_down_world_ = nullptr;
    PointCloudXYZICuda* d_feats_down_world_ = nullptr;       
    pointWithVarCuda* h_input_points_ = nullptr;
    pointWithVarCuda* d_input_points_ = nullptr;             
    
    int feats_down_world_size_ = 0;
    int effect_feat_num_ = 0;
    double extR_[9], extT_[3];
    float build_residual_time, ekf_time;
    float ave_build_residual_time = 0.0;
    float ave_ekf_time = 0.0;
    int scan_count = 0;
    StatesGroupCuda state_;
    Eigen::Vector3d position_last_;
    Eigen::Vector3d last_slide_position = {0,0,0};
    geometry_msgs::Quaternion geoQuat_;
    int feats_down_size_;
    int effct_feat_num_;
    
    Mat33* d_cross_mat_list_ = nullptr;
    Mat33* d_body_cov_list_ = nullptr;
    pointWithVarCuda* d_pv_list_ = nullptr;
    PointToPlaneCuda* d_ptpl_list_ = nullptr;
    PointCloudXYZICuda* d_world_lidar_ = nullptr;
    PointToPlaneCuda* d_all_ptpl_list_ = nullptr;
    bool* d_useful_ptpl_ = nullptr;

    pointWithVarCuda* h_pv_list_ = nullptr;
    PointToPlaneCuda* h_ptpl_list_ = nullptr;
    PointToPlaneCuda* h_all_ptpl_list_ = nullptr;
    bool* h_useful_ptpl_ = nullptr;

    double* d_Hsub_;
    double* d_Hsub_T_R_inv_;
    double* d_R_inv_;
    double* d_meas_vec_;

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status;
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverStatus_t cusolver_status;
    cudaStream_t stream0, stream1;

    int iteration = 0;

    VoxelMapManagerCuda(VoxelMapConfigCuda &config_setting_cuda)
            : config_setting_cuda_(config_setting_cuda)
    {
        current_frame_id_ = 0;
        d_feats_undistort_ = nullptr;
        d_feats_down_body_ = nullptr;
        d_feats_down_world_ = nullptr;
    }
    
    __host__ void VoxelMapMalloc();
    __host__ void VoxelMapRelease();
    __host__ void BuildVoxelMapCuda();
    __host__ void StateEstimationCuda(StatesGroupCuda &h_state_propagate);
    __host__ void BuildResidualListOMPCuda();
    __host__ void UpdateVoxelMapCuda();
    __host__ void PVListUpdateCuda();
    __host__ void mapSlidingCuda();
    __host__ void VoxelMemCpy(int num);
    __host__ void PvListCpy(int num);
    __host__ void Log(const double* R, double* out);
    __host__ void Exp(const double v1, const double v2, const double v3, double* out);
    __host__ void HandleCreate();
    __host__ void HandleDestroy();
    __host__ void StreamCreate();
};
typedef std::shared_ptr<VoxelMapManagerCuda> VoxelMapManagerCudaPtr;

inline void eigenToCuda(Eigen::Vector3d& src, double* dst, int index)
{
    Eigen::Map<Eigen::Vector3d>(dst + index) = src;
}

inline void eigenToCuda(Eigen::Matrix3d& src, double* dst, int index)
{
    Eigen::Map<Eigen::Matrix3d>(dst + index) = src;
}

inline void eigenToCuda(Eigen::Matrix<double, 6, 6>& src, double* dst, int index)
{
    Eigen::Map<Eigen::Matrix<double, 6, 6>>(dst + index) = src;
}

inline void eigenToCuda(Eigen::Matrix<double, 19, 19>& src, double*dst, int index)
{
    Eigen::Map<Eigen::Matrix<double, 19, 19>>(dst + index) = src;
}

inline void doubleToEigen(double* src, Eigen::Vector3d& dst, int index)
{
    dst = Eigen::Map<const Eigen::Vector3d>(src + index);
}

inline void doubleToEigen(double* src, Eigen::Matrix3d& dst, int index)
{
    dst = Eigen::Map<const Eigen::Matrix3d>(src + index);
}

inline void doubleToEigen(double* src, Eigen::Matrix<double, 6, 1>& dst, int index)
{
    dst = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(src + index);
}

inline void doubleToEigen(double* src, Eigen::Matrix<double, 19, 19>& dst, int index)
{
    dst = Eigen::Map<const Eigen::Matrix<double, 19, 19>>(src + index);
}

inline void eigenToCuda(Eigen::Vector3d& src, double* dst, int index = 0);
inline void eigenToCuda(Eigen::Matrix3d& src, double* dst, int index = 0);
inline void eigenToCuda(Eigen::Matrix<double, 6, 6>& src, double* dst, int index = 0);
inline void eigenToCuda(Eigen::Matrix<double, 19, 19>& src, double*dst, int index = 0);
inline void doubleToEigen(double* src, Eigen::Vector3d& dst, int index = 0);
inline void doubleToEigen(double* src, Eigen::Matrix3d& dst, int index = 0);
inline void doubleToEigen(double* src, Eigen::Matrix<double, 6, 1>& dst, int index = 0);
inline void doubleToEigen(double* src, Eigen::Matrix<double, 19, 19>& dst, int index = 0);


#endif