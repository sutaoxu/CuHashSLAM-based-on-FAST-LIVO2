/*
This file is part of CuHashSLAM
*/

#include <vector>
#include "utils/types.h"
#include "voxel_map_cuda.h"
#include <Eigen/Dense>
#include <cuco/static_map.cuh>

#ifndef DEG2RAD
#define DEG2RAD(x) ((x)*0.017453293)
#endif

#ifndef RAD2DEG
#define RAD2DEG(x) ((x)*57.29578)
#endif

__managed__ double h_d_rot_var[9];
__managed__ double h_d_t_var[9];
__managed__ double h_d_HTz[6];
__managed__ double h_d_Hsub_T_R_inv_Hsub[36];
// __managed__ int free_stack_root[TOTAL_CAPACITY];
// __managed__ int free_stack_leaf[TOTAL_CAPACITY];
// __managed__ int free_bottom_root = 0;
// __managed__ int free_bottom_leaf = 0;
// __managed__ int max_index_root = 0;
// __managed__ int max_index_leaf = 0;
// __managed__ int erase_count_root = 0;
// __managed__ int erase_count_leaf = 0;

// __managed__ VOXEL_LOCATION_CUDA d_root_voxel_location_cuda[TOTAL_CAPACITY];      
// __managed__ RootVoxelCuda d_root_voxel_cuda[TOTAL_CAPACITY];                     
// __managed__ VOXEL_LOCATION_CUDA d_leaf_voxel_location_cuda[TOTAL_CAPACITY];      
// __managed__ LeafVoxelCuda d_leaf_voxel_cuda[TOTAL_CAPACITY];

__device__ int free_bottom_root = 0;
__device__ int free_bottom_leaf = 0;
__device__ int free_bottom_root_count = 0;
__device__ int free_bottom_leaf_count = 0;
__device__ int erase_count_root = 0;
__device__ int erase_count_leaf = 0;

__device__ int free_stack_root[TOTAL_CAPACITY];
__device__ int free_stack_leaf[TOTAL_CAPACITY];

__device__ VOXEL_LOCATION_CUDA* root_voxel_erased[5000];
__device__ VOXEL_LOCATION_CUDA* leaf_voxel_erased[5000];

int max_index_root = 0;
int max_index_leaf = 0;
int h_free_bottom_root = 0;
int h_free_bottom_leaf = 0;

auto root_voxel_map_cuda = cuco::static_map{cuco::extent<std::size_t, TOTAL_CAPACITY>{},
                                            cuco::empty_key<VOXEL_LOCATION_CUDA*>{ROOT_VOXEL_EMPTY_KEY},
                                            cuco::empty_value<RootVoxelCuda*>{ROOT_VOXEL_EMPTY_VALUE},
                                            VOXEL_LOCATION_CUDA_EQUAL{},
                                            cuco::linear_probing<1, VOXEL_LOCATION_CUDA_HASH>{}};

auto leaf_voxel_map_cuda = cuco::static_map{cuco::extent<std::size_t, TOTAL_CAPACITY>{},
                                            cuco::empty_key<VOXEL_LOCATION_CUDA*>{LEAF_VOXEL_EMPTY_KEY},
                                            cuco::empty_value<LeafVoxelCuda*>{LEAF_VOXEL_EMPTY_VALUE},
                                            VOXEL_LOCATION_CUDA_EQUAL{},
                                            cuco::linear_probing<1, VOXEL_LOCATION_CUDA_HASH>{}};


__device__ void calcBodyCovCuda(double *pb, const double range_inc, const double degree_inc, double *cov)
{
    if (pb[2] == 0.0)  {pb[2] = 0.0001;}
    float range = sqrt(pb[0]*pb[0] + pb[1]*pb[1] + pb[2]*pb[2]);
    float range_var = range_inc * range_inc;
    double direction_var[4] = {pow(sin(DEG2RAD(degree_inc)), 2.0), 0.0, 0.0, pow(sin(DEG2RAD(degree_inc)), 2.0)};
    double direction[3] = {pb[0]/range, pb[1]/range, pb[2]/range};
    double direction_hat[9] = {0.0, -direction[2], direction[1], direction[2], 0.0, -direction[0], -direction[1], direction[0], 0.0};
    double base_vector1[3] = {1.0, 1.0, -(direction[0]+direction[1]) / direction[2]};
    double base_vector1_range = sqrt(base_vector1[0]*base_vector1[0] + base_vector1[1]*base_vector1[1] + base_vector1[2]*base_vector1[2]);
    base_vector1[0] /= base_vector1_range; base_vector1[1] /= base_vector1_range; base_vector1[2] /= base_vector1_range;
    double base_vector2[3] = {base_vector1[1]*direction[2] - base_vector1[2]*direction[1],
                              base_vector1[2]*direction[0] - base_vector1[0]*direction[2],
                              base_vector1[0]*direction[1] - base_vector1[1]*direction[0]};
    double base_vector2_range = sqrt(base_vector2[0]*base_vector2[0] + base_vector2[1]*base_vector2[1] + base_vector2[2]*base_vector2[2]);
    base_vector2[0] /= base_vector2_range; base_vector2[1] /= base_vector2_range; base_vector2[2] /= base_vector2_range;
    double N[6] = {base_vector1[0], base_vector2[0], base_vector1[1], base_vector2[1], base_vector1[2], base_vector2[2]};
    double A[6];
    A[0] = range * (direction_hat[0]*N[0] + direction_hat[1]*N[2] + direction_hat[2]*N[4]);
    A[1] = range * (direction_hat[0]*N[1] + direction_hat[1]*N[3] + direction_hat[2]*N[5]);
    A[2] = range * (direction_hat[3]*N[0] + direction_hat[4]*N[2] + direction_hat[5]*N[4]);
    A[3] = range * (direction_hat[3]*N[1] + direction_hat[4]*N[3] + direction_hat[5]*N[5]);
    A[4] = range * (direction_hat[6]*N[0] + direction_hat[7]*N[2] + direction_hat[8]*N[4]);
    A[5] = range * (direction_hat[6]*N[1] + direction_hat[7]*N[3] + direction_hat[8]*N[5]);

    double temp[6];
    temp[0] = A[0] * direction_var[0] + A[1] * direction_var[2];
    temp[1] = A[0] * direction_var[1] + A[1] * direction_var[3];
    temp[2] = A[2] * direction_var[0] + A[3] * direction_var[2];
    temp[3] = A[2] * direction_var[1] + A[3] * direction_var[3];
    temp[4] = A[4] * direction_var[0] + A[5] * direction_var[2];
    temp[5] = A[4] * direction_var[1] + A[5] * direction_var[3];

    cov[0] = range_var * direction[0] * direction[0] + temp[0] * A[0] + temp[1] * A[1];
    cov[1] = range_var * direction[0] * direction[1] + temp[0] * A[2] + temp[1] * A[3];
    cov[2] = range_var * direction[0] * direction[2] + temp[0] * A[4] + temp[1] * A[5];
    cov[3] = range_var * direction[1] * direction[0] + temp[2] * A[0] + temp[3] * A[1];
    cov[4] = range_var * direction[1] * direction[1] + temp[2] * A[2] + temp[3] * A[3];
    cov[5] = range_var * direction[1] * direction[2] + temp[2] * A[4] + temp[3] * A[5];
    cov[6] = range_var * direction[2] * direction[0] + temp[4] * A[0] + temp[5] * A[1];
    cov[7] = range_var * direction[2] * direction[1] + temp[4] * A[2] + temp[5] * A[3];
    cov[8] = range_var * direction[2] * direction[2] + temp[4] * A[4] + temp[5] * A[5];
}


__host__ __device__ void fill_array(double* data, int N, double value) 
{
    for (int i = 0; i < N; ++i) {
        data[i] = value;
    }
}

__device__ void normalize3(double v[3]) {

    double len2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];

    if (len2 < 1e-12) return;
    double invLen = rsqrt(len2); // faster than 1.0/sqrt(len2)
    v[0] *= invLen;
    v[1] *= invLen;
    v[2] *= invLen;
}

__device__ void eigenDecompose33 (const double A[9], double* evalMin, double* evalMid, double* evalMax,
                                  double evecMin[3], double evecMid[3], double evecMax[3])
{
    double m11 = A[0], m12 = A[1], m13 = A[2];
    double m22 = A[4], m23 = A[5];
    double m33 = A[8];

    double p1 = m12*m12 + m13*m13 + m23*m23;
    if (p1 < 1e-12)
    {
        double lm[3] = { m11, m22, m33 };
        int idxMin = 0, idxMax = 0;
        if (lm[1] < lm[idxMin]) idxMin = 1;
        if (lm[2] < lm[idxMin]) idxMin = 2;
        if (lm[1] > lm[idxMax]) idxMax = 1;
        if (lm[2] > lm[idxMax]) idxMax = 2;
        int idxMid = 3 - idxMin - idxMax;
        *evalMin = lm[idxMin];
        *evalMid = lm[idxMid];
        *evalMax = lm[idxMax];

        evecMin[0] = (idxMin==0); evecMin[1] = (idxMin==1); evecMin[2] = (idxMin==2);
        evecMid[0] = (idxMid==0); evecMid[1] = (idxMid==1); evecMid[2] = (idxMid==2);
        evecMax[0] = (idxMax==0); evecMax[1] = (idxMax==1); evecMax[2] = (idxMax==2);
        return;
    }

    double q = (m11 + m22 + m33) * (1.0/3.0);
    double a11 = m11 - q, a22 = m22 - q, a33 = m33 - q;
    double p2 = a11*a11 + a22*a22 + a33*a33 + 2.0*p1;
    double p = sqrt(p2 * (1.0/6.0));

    double B11 = a11 / p;
    double B22 = a22 / p;
    double B33 = a33 / p;
    double B12 = m12 / p;
    double B13 = m13 / p;
    double B23 = m23 / p;

    double detB = B11*(B22*B33 - B23*B23) - B12*(B12*B33 - B23*B13) + B13*(B12*B23 - B22*B13);
    double r = detB * 0.5;

    r = r < -1.0 ? -1.0 : (r > 1.0 ? 1.0 : r);
    double phi = acos(r) * (1.0/3.0);
    double c0  = cos(phi);
    double c1  = cos(phi + 2.0*CUDART_PI/3.0);

    double lam0 = q + 2.0 * p * c0;
    double lam2 = q + 2.0 * p * c1;
    double lam1 = 3.0*q - lam0 - lam2;

    double lam[3] = { lam0, lam1, lam2 };
    double evecs[9];
    for (int i = 0; i < 3; ++i) 
    {
        double m1 = A[0] - lam[i];
        double m2 = A[1],     m3 = A[2];
        double m4 = A[3];
        double m5 = A[4] - lam[i];
        double m6 = A[5];
        double m7 = A[6];
        double m8 = A[7];
        double m9 = A[8] - lam[i];

        double x = m2*m6 - m3*m5;
        double y = m3*m4 - m1*m6;
        double z = m1*m5 - m2*m4;
        double norm = sqrt(x*x + y*y + z*z);
        if (norm < 1e-8) 
        {
            x = m2*m9 - m3*m8;
            y = m3*m7 - m1*m9;
            z = m1*m8 - m2*m7;
            norm = sqrt(x*x + y*y + z*z);
        }
        double invn = 1.0 / (norm < 1e-12 ? 1.0 : norm);
        evecs[i*3 + 0] = x * invn;
        evecs[i*3 + 1] = y * invn;
        evecs[i*3 + 2] = z * invn;
    }

    int idxMin = 0, idxMax = 0;
    if (lam[1] < lam[idxMin]) idxMin = 1;
    if (lam[2] < lam[idxMin]) idxMin = 2;
    if (lam[1] > lam[idxMax]) idxMax = 1;
    if (lam[2] > lam[idxMax]) idxMax = 2;
    int idxMid = 3 - idxMin - idxMax;

    *evalMin = lam[idxMin];
    *evalMid = lam[idxMid];
    *evalMax = lam[idxMax];
    
    for (int k = 0; k < 3; ++k) 
    {
        evecMin[k] = evecs[idxMin*3 + k];
        evecMid[k] = evecs[idxMid*3 + k];
        evecMax[k] = evecs[idxMax*3 + k];
    }

    normalize3(evecMin);
    normalize3(evecMax);
}

__device__ void init_plane_cuda(const pointWithVarCuda *d_points, int points_size, VoxelPlaneCuda *d_plane, float planer_threshold)
{
    fill_array(d_plane->plane_var_, 36, 0.0);
    fill_array(d_plane->covariance_, 9, 0.0);
    fill_array(d_plane->center_, 3, 0.0);
    fill_array(d_plane->normal_, 3, 0.0);
    d_plane->points_size_ = points_size;
    d_plane->radius_ = 0.0f;
    for (int i=0; i<points_size; i++)
    {
        double point_w[3] = {0.0,0.0,0.0};
        double temp[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        for (int j=0; j<3; j++)  {point_w[j] = d_points[i].point_w[j];}
        for (int m=0; m<3; m++)
        {
            for (int n=0; n<3; n++)  {temp[3*m+n] = point_w[m] * point_w[n];}
        }
        for (int j=0; j<9; j++)  {d_plane->covariance_[j] += temp[j];}
        for (int j=0; j<3; j++)  {d_plane->center_[j] += point_w[j];}
    }

    for (int i=0; i<3; i++)  {d_plane->center_[i] = d_plane->center_[i] / d_plane->points_size_;}
    for (int i=0; i<9; i++)  {d_plane->covariance_[i] = d_plane->covariance_[i] / d_plane->points_size_;}
    double temp[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    for (int m=0; m<3; m++)
    {
        for (int n=0; n<3; n++)  {temp[3*m+n] = d_plane->center_[m] * d_plane->center_[n];}
    }
    for (int i=0; i<9; i++)  {d_plane->covariance_[i] = d_plane->covariance_[i] - temp[i];}

    double evalMin=0.0, evalMid=0.0, evalMax=0.0;
    double evecMin[3]={0.0,0.0,0.0}, evecMid[3]={0.0,0.0,0.0}, evecMax[3]={0.0,0.0,0.0};
    
    eigenDecompose33(d_plane->covariance_, &evalMin, &evalMid, &evalMax, evecMin, evecMid, evecMax);

    double J_Q[9] = {1.0/(d_plane->points_size_), 0.0, 0.0, 
                     0.0, 1.0/(d_plane->points_size_), 0.0,
                     0.0, 0.0, 1.0/(d_plane->points_size_)};
    if (evalMin < planer_threshold)
    {
        for (int i=0; i<points_size; i++)
        {
            double J[18];
            double F[9];
            double temp1[3];
            double temp2[9];
            double temp3[18];
            double temp4[36];
            double F_m[3] = {0.0,0.0,0.0};
            double point_w[3];
            double var[9];
            for (int j=0; j<3; j++)  {point_w[j] = d_points[i].point_w[j];}
            for (int j=0; j<9; j++)  {var[j] = d_points[i].var[j];}
            F[0] = 0.0; F[1] = 0.0; F[2] = 0.0;
            
            for (int j=0; j<3; j++)
            {
                temp1[j] = (point_w[j] - d_plane->center_[j]) / ((d_plane->points_size_) * (evalMin - evalMid));
            }
            for (int m=0; m<3; m++)
            {
                for (int n=0; n<3; n++)  {temp2[3*m+n] = evecMid[m] * evecMin[n] + evecMin[m] * evecMid[n];}
            }
            for (int j=0; j<3; j++)
            {
                F_m[0] += temp1[j]*temp2[3*j+0];  F_m[1] += temp1[j]*temp2[3*j+1];  F_m[2] += temp1[j]*temp2[3*j+2];
            }
            F[3] = F_m[0]; F[4] = F_m[1]; F[5] = F_m[2];
            
            F_m[0] = 0; F_m[1] = 0; F_m[2] = 0;
            for (int j=0; j<3; j++)
            {
                temp1[j] = (point_w[j] - d_plane->center_[j]) / ((d_plane->points_size_) * (evalMin - evalMax));
            }
            for (int m=0; m<3; m++)
            {
                for (int n=0; n<3; n++)  {temp2[3*m+n] = evecMax[m] * evecMin[n] + evecMin[m] * evecMax[n];}
            }
            for (int j=0; j<3; j++)
            {
                F_m[0] += temp1[j]*temp2[3*j+0];  F_m[1] += temp1[j]*temp2[3*j+1];  F_m[2] += temp1[j]*temp2[3*j+2];
            }
            F[6] = F_m[0]; F[7] = F_m[1]; F[8] = F_m[2];
            
            for (int j=0; j<3; j++)
            {
                J[3*j+0] = evecMin[j]*F[0] + evecMid[j]*F[3] + evecMax[j]*F[6];
                J[3*j+1] = evecMin[j]*F[1] + evecMid[j]*F[4] + evecMax[j]*F[7];
                J[3*j+2] = evecMin[j]*F[2] + evecMid[j]*F[5] + evecMax[j]*F[8];
                J[3*(j+3)+0] = J_Q[3*j+0];  J[3*(j+3)+1] = J_Q[3*j+1];  J[3*(j+3)+2] = J_Q[3*j+2];
            }

            fill_array(temp3, 18, 0.0);
            fill_array(temp4, 36, 0.0);
            for (int m=0; m<6; m++)
            {
                for (int n=0; n<3; n++)
                {
                    double sum = 0.0;
                    for(int k=0; k<3; k++)  {sum += J[3*m+k] * var[3*k+n];}
                    temp3[3*m+n] = sum;
                }
            }

            for (int m=0; m<6; m++)
            {
                for (int n=0; n<6; n++)
                {
                    double sum = 0.0;
                    for(int k=0; k<3; k++)  {sum += temp3[3*m+k] * J[3*n+k];}
                    temp4[6*m+n] = sum;
                }
            }

            for (int j=0; j<36; j++)  {d_plane->plane_var_[j] += temp4[j];}
        }

        for (int i=0; i<3; i++)
        {
            d_plane->normal_[i] = evecMin[i]; d_plane->y_normal_[i] = evecMid[i]; d_plane->x_normal_[i] = evecMax[i];
        }
        d_plane->min_eigen_value_ = evalMin; d_plane->mid_eigen_value_ = evalMid; d_plane->max_eigen_value_ = evalMax;
        d_plane->radius_ = sqrt(evalMax);
        d_plane->d_ = -(d_plane->normal_[0] * d_plane->center_[0] + d_plane->normal_[1] * d_plane->center_[1] 
                        + d_plane->normal_[2] * d_plane->center_[2]);
        d_plane->is_plane_ = true;  d_plane->is_update_ = true;
        if (!d_plane->is_init_)
        {
            d_plane->id_ = atomicAdd(&voxel_plane_cuda_id, 1);
            d_plane->is_init_ = true;
        }
    }
    else
    {
        d_plane->is_update_ = true;
        d_plane->is_plane_ = false;
    }
}

template <typename FindRefLeaf, typename InsertRefLeaf>
__device__ void init_octo_tree_root_cuda(VOXEL_LOCATION_CUDA* root_location, RootVoxelCuda* root_value,
                                    FindRefLeaf find_ref_leaf, InsertRefLeaf insert_ref_leaf,
                                    VOXEL_LOCATION_CUDA* d_leaf_voxel_location_cuda, LeafVoxelCuda* d_leaf_voxel_cuda)
{
    if (root_value->new_points_ > root_value->points_size_threshold_)
    {
        init_plane_cuda(root_value->temp_points_, root_value->new_points_, &(root_value->plane_), root_value->planer_threshold_);
        if (root_value->plane_.is_plane_ == true)  
        {
            root_value->leaf_enable_ = 0;
            if (root_value->new_points_ >= 50)  
            {
                root_value->new_points_ = 0;
                root_value->update_enable_ = 0;
            }
        }
        else
        {
            root_value->leaf_enable_ = 1;
            cut_octo_tree_cuda(root_location, root_value, find_ref_leaf, insert_ref_leaf, 
                               d_leaf_voxel_location_cuda, d_leaf_voxel_cuda);
        }
        root_value->init_octo_ = 1;
        // root_value->new_points_ = 0;
    }
}

template <typename FindRefLeaf, typename InsertRefLeaf>
__device__ void cut_octo_tree_cuda(VOXEL_LOCATION_CUDA* root_location, RootVoxelCuda* root_value,
                                   FindRefLeaf find_ref_leaf, InsertRefLeaf insert_ref_leaf,
                                   VOXEL_LOCATION_CUDA* d_leaf_voxel_location_cuda, LeafVoxelCuda* d_leaf_voxel_cuda)
{
    for (int i=0; i<(root_value->new_points_); i++)
    {
        int xyz[3] = {0, 0, 0};
        if (root_value->temp_points_[i].point_w[0] > root_value->voxel_center_[0])  xyz[0] = 1;
        if (root_value->temp_points_[i].point_w[1] > root_value->voxel_center_[1])  xyz[1] = 1;
        if (root_value->temp_points_[i].point_w[2] > root_value->voxel_center_[2])  xyz[2] = 1;
        int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
        VOXEL_LOCATION_CUDA leaf_voxel_position = root_value->leaf_voxel_[leafnum];
        auto iter = find_ref_leaf.find(&leaf_voxel_position);
        if (iter != find_ref_leaf.end())
        {
            int point_index = atomicAdd(&(iter->second->new_points_), 1);
            if (iter->second->new_points_ >= 50)
            {
                atomicExch(&(iter->second->new_points_), 50);
                atomicExch(&(iter->second->update_enable_), 0);
                return;
            }
            iter->second->temp_points_[point_index] = root_value->temp_points_[i];
        }
        else
        {
            int current_free_bottom_leaf = atomicAdd(&free_bottom_leaf, 1);
            int current_leaf_voxel_index = free_stack_leaf[current_free_bottom_leaf];
            if (current_leaf_voxel_index >= TOTAL_CAPACITY)  return;
            d_leaf_voxel_location_cuda[current_leaf_voxel_index] = root_value->leaf_voxel_[leafnum];
            d_leaf_voxel_cuda[current_leaf_voxel_index].voxel_center_[0] = root_value->voxel_center_[0] + (2*xyz[0]-1) * (root_value->quater_length_);
            d_leaf_voxel_cuda[current_leaf_voxel_index].voxel_center_[1] = root_value->voxel_center_[1] + (2*xyz[1]-1) * (root_value->quater_length_);
            d_leaf_voxel_cuda[current_leaf_voxel_index].voxel_center_[2] = root_value->voxel_center_[2] + (2*xyz[2]-1) * (root_value->quater_length_);
            d_leaf_voxel_cuda[current_leaf_voxel_index].parent_voxel_ = *root_location;
            for (int m=0; m<5; m++)  {d_leaf_voxel_cuda[current_leaf_voxel_index].layer_init_num_[m] = root_value->layer_init_num_[m];}
            d_leaf_voxel_cuda[current_leaf_voxel_index].points_size_threshold_ = root_value->layer_init_num_[1];
            d_leaf_voxel_cuda[current_leaf_voxel_index].update_size_threshold_ = 5;
            d_leaf_voxel_cuda[current_leaf_voxel_index].new_points_ = 0;
            d_leaf_voxel_cuda[current_leaf_voxel_index].quater_length_ = (root_value->quater_length_) / 2;
            d_leaf_voxel_cuda[current_leaf_voxel_index].planer_threshold_ = root_value->planer_threshold_;
            d_leaf_voxel_cuda[current_leaf_voxel_index].update_enable_ = 1;
            d_leaf_voxel_cuda[current_leaf_voxel_index].init_octo_ = 0;
            d_leaf_voxel_cuda[current_leaf_voxel_index].is_valid_ = 0;

            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.radius_ = 0.0f;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.min_eigen_value_ = 1.0f;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.mid_eigen_value_ = 1.0f;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.max_eigen_value_ = 1.0f;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.d_ = 0.0f;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.points_size_ = 0;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.id_ = 0;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.is_plane_ = false;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.is_init_ = false;
            d_leaf_voxel_cuda[current_leaf_voxel_index].plane_.is_update_ = false;

            int point_index = atomicAdd(&(d_leaf_voxel_cuda[current_leaf_voxel_index].new_points_), 1);
            d_leaf_voxel_cuda[current_leaf_voxel_index].temp_points_[point_index] = root_value->temp_points_[i];
            bool is_inserted = insert_ref_leaf.insert(cuco::pair{&d_leaf_voxel_location_cuda[current_leaf_voxel_index], 
                                                                &d_leaf_voxel_cuda[current_leaf_voxel_index]});
            if (is_inserted)  d_leaf_voxel_cuda[current_leaf_voxel_index].is_valid_ = 1;
            else
            {
                d_leaf_voxel_cuda[current_leaf_voxel_index].is_valid_ = 0;
                auto iter1 = find_ref_leaf.find(&leaf_voxel_position);
                int point_index = atomicAdd(&(iter1->second->new_points_), 1);
                if (point_index >= 50)
                {
                    atomicExch(&(iter1->second->new_points_), 50);
                    atomicExch(&(iter1->second->update_enable_), 0);
                    return;
                }
                iter1->second->temp_points_[point_index] = root_value->temp_points_[i];
            }
        }
    }
}

__device__ void init_octo_tree_leaf_cuda(VOXEL_LOCATION_CUDA* leaf_location, LeafVoxelCuda* leaf_value)
{
    if (leaf_value->new_points_ > leaf_value->points_size_threshold_)
    {
        init_plane_cuda(leaf_value->temp_points_, leaf_value->new_points_, &(leaf_value->plane_), leaf_value->planer_threshold_);
        if (leaf_value->new_points_ >= 50)  
        {
            leaf_value->new_points_ = 0;
            leaf_value->update_enable_ = 0;
        }
        leaf_value->init_octo_ = true;
        // leaf_value->new_points_ = 0;
    }
}

__device__ void build_single_residual_leaf_cuda(pointWithVarCuda &pv, LeafVoxelCuda* leaf_value, double &prob, bool &is_success,
                                           PointToPlaneCuda &single_ptpl, double sigma_num)
{
    double radius_k = 3.0;
    double p_w[3] = {pv.point_w[0], pv.point_w[1], pv.point_w[2]};
    if (leaf_value->plane_.is_plane_)
    {
        VoxelPlaneCuda &plane = leaf_value->plane_;
        double p_world_to_center[3] = {p_w[0]-plane.center_[0], p_w[1]-plane.center_[1], p_w[2]-plane.center_[2]};
        float dis_to_plane = fabs(plane.normal_[0]*p_w[0] + plane.normal_[1]*p_w[1] + plane.normal_[2]*p_w[2] + plane.d_);
        float dis_to_center = (plane.center_[0]-p_w[0]) * (plane.center_[0]-p_w[0]) + (plane.center_[1]-p_w[1]) *
                               (plane.center_[1]-p_w[1]) + (plane.center_[2]-p_w[2]) * (plane.center_[2]-p_w[2]);
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_center);

        if (range_dis <= radius_k * plane.radius_)
        {
            double J_nq[6];
            J_nq[0] = p_w[0] - plane.center_[0];  J_nq[1] = p_w[1] - plane.center_[1];  J_nq[2] = p_w[2] - plane.center_[2];
            J_nq[3] = -plane.normal_[0];  J_nq[4] = -plane.normal_[1];  J_nq[5] = -plane.normal_[2];
            
            double temp1[6], temp2[3];
            temp1[0] = J_nq[0]*plane.plane_var_[0] + J_nq[1]*plane.plane_var_[6] + J_nq[2]*plane.plane_var_[12] + 
                      J_nq[3]*plane.plane_var_[18] + J_nq[4]*plane.plane_var_[24] + J_nq[5]*plane.plane_var_[30];
            temp1[1] = J_nq[0]*plane.plane_var_[1] + J_nq[1]*plane.plane_var_[7] + J_nq[2]*plane.plane_var_[13] + 
                      J_nq[3]*plane.plane_var_[19] + J_nq[4]*plane.plane_var_[25] + J_nq[5]*plane.plane_var_[31];
            temp1[2] = J_nq[0]*plane.plane_var_[2] + J_nq[1]*plane.plane_var_[8] + J_nq[2]*plane.plane_var_[14] + 
                      J_nq[3]*plane.plane_var_[20] + J_nq[4]*plane.plane_var_[26] + J_nq[5]*plane.plane_var_[32];
            temp1[3] = J_nq[0]*plane.plane_var_[3] + J_nq[1]*plane.plane_var_[9] + J_nq[2]*plane.plane_var_[15] + 
                      J_nq[3]*plane.plane_var_[21] + J_nq[4]*plane.plane_var_[27] + J_nq[5]*plane.plane_var_[33];
            temp1[4] = J_nq[0]*plane.plane_var_[4] + J_nq[1]*plane.plane_var_[10] + J_nq[2]*plane.plane_var_[16] + 
                      J_nq[3]*plane.plane_var_[22] + J_nq[4]*plane.plane_var_[28] + J_nq[5]*plane.plane_var_[34];
            temp1[5] = J_nq[0]*plane.plane_var_[5] + J_nq[1]*plane.plane_var_[11] + J_nq[2]*plane.plane_var_[17] + 
                      J_nq[3]*plane.plane_var_[23] + J_nq[4]*plane.plane_var_[29] + J_nq[5]*plane.plane_var_[35];
            double sigma_l = temp1[0]*J_nq[0] + temp1[1]*J_nq[1] + temp1[2]*J_nq[2] + 
                             temp1[3]*J_nq[3] + temp1[4]*J_nq[4] + temp1[5]*J_nq[5];

            temp2[0] = plane.normal_[0]*pv.var[0] + plane.normal_[1]*pv.var[3] + plane.normal_[2]*pv.var[6];
            temp2[1] = plane.normal_[0]*pv.var[1] + plane.normal_[1]*pv.var[4] + plane.normal_[2]*pv.var[7];
            temp2[2] = plane.normal_[0]*pv.var[2] + plane.normal_[1]*pv.var[5] + plane.normal_[2]*pv.var[8];
            sigma_l += temp2[0]*plane.normal_[0] + temp2[1]*plane.normal_[1] + temp2[2]*plane.normal_[2];
            
            if (dis_to_plane < sigma_num * sqrt(sigma_l))
            {
                is_success = true;
                double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
                if (this_prob > prob)
                {
                    prob = this_prob;
                    pv.normal[0] = plane.normal_[0]; pv.normal[1] = plane.normal_[1]; pv.normal[2] = plane.normal_[2];
                    for (int i=0; i<3; i++)  single_ptpl.point_b_[i] = pv.point_b[i];
                    for (int i=0; i<3; i++)  single_ptpl.point_w_[i] = pv.point_w[i];
                    for (int i=0; i<3; i++)  single_ptpl.normal_[i] = plane.normal_[i];
                    for (int i=0; i<3; i++)  single_ptpl.center_[i] = plane.center_[i];
                    for (int i=0; i<36; i++)  single_ptpl.plane_var_[i] = plane.plane_var_[i];
                    for (int i=0; i<9; i++)  single_ptpl.body_cov_[i] = pv.body_var[i];
                    single_ptpl.d_ = plane.d_;
                    single_ptpl.dis_to_plane_ = plane.normal_[0]*p_w[0] + plane.normal_[1]*p_w[1] + plane.normal_[2]*p_w[2] + plane.d_;
                }
                return;
            }
            else return;
        }
        else return;
    }
}


template<typename FindRefLeaf>
__device__ void build_single_residual_root_cuda(pointWithVarCuda &pv, RootVoxelCuda* root_value, double &prob, bool &is_success,
                                           PointToPlaneCuda &single_ptpl, double sigma_num, FindRefLeaf find_ref_leaf, int x)
{
    double radius_k = 3.0;
    double p_w[3] = {pv.point_w[0], pv.point_w[1], pv.point_w[2]};
    if (root_value->plane_.is_plane_)
    {
        VoxelPlaneCuda &plane = root_value->plane_;
        double p_world_to_center[3] = {p_w[0]-plane.center_[0], p_w[1]-plane.center_[1], p_w[2]-plane.center_[2]};
        float dis_to_plane = fabs(plane.normal_[0]*p_w[0] + plane.normal_[1]*p_w[1] + plane.normal_[2]*p_w[2] + plane.d_);
        float dis_to_center = (plane.center_[0]-p_w[0]) * (plane.center_[0]-p_w[0]) + (plane.center_[1]-p_w[1]) *
                               (plane.center_[1]-p_w[1]) + (plane.center_[2]-p_w[2]) * (plane.center_[2]-p_w[2]);
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_center);

        if (range_dis <= radius_k * plane.radius_)
        {
            double J_nq[6];
            J_nq[0] = p_w[0] - plane.center_[0];  J_nq[1] = p_w[1] - plane.center_[1];  J_nq[2] = p_w[2] - plane.center_[2];
            J_nq[3] = -plane.normal_[0];  J_nq[4] = -plane.normal_[1];  J_nq[5] = -plane.normal_[2];
            
            double temp1[6], temp2[3];
            temp1[0] = J_nq[0]*plane.plane_var_[0] + J_nq[1]*plane.plane_var_[6] + J_nq[2]*plane.plane_var_[12] + 
                      J_nq[3]*plane.plane_var_[18] + J_nq[4]*plane.plane_var_[24] + J_nq[5]*plane.plane_var_[30];
            temp1[1] = J_nq[0]*plane.plane_var_[1] + J_nq[1]*plane.plane_var_[7] + J_nq[2]*plane.plane_var_[13] + 
                      J_nq[3]*plane.plane_var_[19] + J_nq[4]*plane.plane_var_[25] + J_nq[5]*plane.plane_var_[31];
            temp1[2] = J_nq[0]*plane.plane_var_[2] + J_nq[1]*plane.plane_var_[8] + J_nq[2]*plane.plane_var_[14] + 
                      J_nq[3]*plane.plane_var_[20] + J_nq[4]*plane.plane_var_[26] + J_nq[5]*plane.plane_var_[32];
            temp1[3] = J_nq[0]*plane.plane_var_[3] + J_nq[1]*plane.plane_var_[9] + J_nq[2]*plane.plane_var_[15] + 
                      J_nq[3]*plane.plane_var_[21] + J_nq[4]*plane.plane_var_[27] + J_nq[5]*plane.plane_var_[33];
            temp1[4] = J_nq[0]*plane.plane_var_[4] + J_nq[1]*plane.plane_var_[10] + J_nq[2]*plane.plane_var_[16] + 
                      J_nq[3]*plane.plane_var_[22] + J_nq[4]*plane.plane_var_[28] + J_nq[5]*plane.plane_var_[34];
            temp1[5] = J_nq[0]*plane.plane_var_[5] + J_nq[1]*plane.plane_var_[11] + J_nq[2]*plane.plane_var_[17] + 
                      J_nq[3]*plane.plane_var_[23] + J_nq[4]*plane.plane_var_[29] + J_nq[5]*plane.plane_var_[35];
            double sigma_l = temp1[0]*J_nq[0] + temp1[1]*J_nq[1] + temp1[2]*J_nq[2] + 
                             temp1[3]*J_nq[3] + temp1[4]*J_nq[4] + temp1[5]*J_nq[5];

            temp2[0] = plane.normal_[0]*pv.var[0] + plane.normal_[1]*pv.var[3] + plane.normal_[2]*pv.var[6];
            temp2[1] = plane.normal_[0]*pv.var[1] + plane.normal_[1]*pv.var[4] + plane.normal_[2]*pv.var[7];
            temp2[2] = plane.normal_[0]*pv.var[2] + plane.normal_[1]*pv.var[5] + plane.normal_[2]*pv.var[8];
            sigma_l += temp2[0]*plane.normal_[0] + temp2[1]*plane.normal_[1] + temp2[2]*plane.normal_[2];
            
            if (dis_to_plane < sigma_num * sqrt(sigma_l))
            {
                is_success = true;
                double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
                if (this_prob > prob)
                {
                    prob = this_prob;
                    pv.normal[0] = plane.normal_[0]; pv.normal[1] = plane.normal_[1]; pv.normal[2] = plane.normal_[2];
                    for (int i=0; i<3; i++)  single_ptpl.point_b_[i] = pv.point_b[i];
                    for (int i=0; i<3; i++)  single_ptpl.point_w_[i] = pv.point_w[i];
                    for (int i=0; i<3; i++)  single_ptpl.normal_[i] = plane.normal_[i];
                    for (int i=0; i<3; i++)  single_ptpl.center_[i] = plane.center_[i];
                    for (int i=0; i<36; i++)  single_ptpl.plane_var_[i] = plane.plane_var_[i];
                    for (int i=0; i<9; i++)  single_ptpl.body_cov_[i] = pv.body_var[i];
                    single_ptpl.d_ = plane.d_;
                    single_ptpl.dis_to_plane_ = plane.normal_[0]*p_w[0] + plane.normal_[1]*p_w[1] + plane.normal_[2]*p_w[2] + plane.d_;
                }
                return;
            }
            else return;
        }
        else return;
    }
    else
    {
        if (root_value->leaf_enable_ == 1)
        {
            for (int i=0; i<8; i++)
            {
                VOXEL_LOCATION_CUDA leaf_location = root_value->leaf_voxel_[i];
                auto iter = find_ref_leaf.find(&leaf_location);
                if (iter != find_ref_leaf.end())
                {
                    LeafVoxelCuda* leaf_value = iter->second;
                    build_single_residual_leaf_cuda(pv, leaf_value, prob, is_success, single_ptpl, sigma_num);
                }
            }
        }
        else return;
    }
}

// template <typename FindRefLeaf>
// __device__ void clear_mem_cuda(FindRefLeaf find_ref_leaf, VOXEL_LOCATION_CUDA* root_location, RootVoxelCuda* root_value, 
//                         int x_max, int x_min, int y_max, int y_min, int z_max, int z_min)
// {
//     bool should_remove = (root_location->x)>x_max || (root_location->x)<x_min || (root_location->y)>y_max || 
//                          (root_location->y)<y_min || (root_location->z)>z_max || (root_location->z)<z_min;
//     if (should_remove)
//     {
//         root_value->is_valid_ = 0;
//         int erase_index_root = atomicAdd(&erase_count_root, 1);
//         root_voxel_erased[erase_index_root] = root_location;
//         if (root_value->leaf_enable_ == 1)
//         {
//             for (int i=0; i<8; i++)
//             {
//                 auto iter = find_ref_leaf.find(&(root_value->leaf_voxel_[i]));
//                 if (iter != find_ref_leaf.end())
//                 {
//                     int erase_index_leaf = atomicAdd(&erase_count_leaf, 1);
//                     leaf_voxel_erased[erase_index_leaf] = iter->first;
//                     iter->second->is_valid_ = 0;
//                 }
//             }
//         }
//     }
// }

__global__ void StateEstimationCudaKernel1(int feats_down_world_size, Mat33* d_cross_mat_list, PointCloudXYZICuda* d_feats_down_body,
                                        Mat33* d_body_cov_list, double dept_err, double beam_err)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  return;

    double point_this[3] = {d_feats_down_body[i].x, d_feats_down_body[i].y, d_feats_down_body[i].z};
    if (point_this[2] == 0)  point_this[2] = 0.001;
    double var[9];
    calcBodyCovCuda(point_this, dept_err, beam_err, var);
    for (int j=0; j<9; j++)  {d_body_cov_list[i].mat[j] = var[j];}
    double temp[3];
    temp[0] = d_ParametersCuda.extR[0]*point_this[0] + d_ParametersCuda.extR[1]*point_this[1] + 
              d_ParametersCuda.extR[2]*point_this[2] + d_ParametersCuda.extT[0];
    temp[1] = d_ParametersCuda.extR[3]*point_this[0] + d_ParametersCuda.extR[4]*point_this[1] + 
              d_ParametersCuda.extR[5]*point_this[2] + d_ParametersCuda.extT[1];
    temp[2] = d_ParametersCuda.extR[6]*point_this[0] + d_ParametersCuda.extR[7]*point_this[1] + 
              d_ParametersCuda.extR[8]*point_this[2] + d_ParametersCuda.extT[2];
    for (int j=0; j<3; j++)  {point_this[j] = temp[j];}
    double point_crossmat[9] = {0.0, -point_this[2], point_this[1], point_this[2], 0.0, -point_this[0], -point_this[1],
                                point_this[0], 0.0};
    for (int j=0; j<9; j++)  {d_cross_mat_list[i].mat[j] = point_crossmat[j];}
}

__global__ void StateEstimationCudaKernel2(int feats_down_world_size, PointCloudXYZICuda* d_feats_down_body, 
                                           PointCloudXYZICuda* d_world_lidar)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  return;

    double p[3] = {d_feats_down_body[i].x, d_feats_down_body[i].y, d_feats_down_body[i].z};
    double temp1[3];
    temp1[0] = d_ParametersCuda.extR[0]*p[0] + d_ParametersCuda.extR[1]*p[1] + d_ParametersCuda.extR[2]*p[2] + d_ParametersCuda.extT[0];
    temp1[1] = d_ParametersCuda.extR[3]*p[0] + d_ParametersCuda.extR[4]*p[1] + d_ParametersCuda.extR[5]*p[2] + d_ParametersCuda.extT[1];
    temp1[2] = d_ParametersCuda.extR[6]*p[0] + d_ParametersCuda.extR[7]*p[1] + d_ParametersCuda.extR[8]*p[2] + d_ParametersCuda.extT[2];

    p[0] = d_state.rot_end[0]*temp1[0] + d_state.rot_end[1]*temp1[1] + d_state.rot_end[2]*temp1[2] + d_state.pos_end[0];
    p[1] = d_state.rot_end[3]*temp1[0] + d_state.rot_end[4]*temp1[1] + d_state.rot_end[5]*temp1[2] + d_state.pos_end[1];
    p[2] = d_state.rot_end[6]*temp1[0] + d_state.rot_end[7]*temp1[1] + d_state.rot_end[8]*temp1[2] + d_state.pos_end[2];

    d_world_lidar[i].x = p[0];  d_world_lidar[i].y = p[1];  d_world_lidar[i].z = p[2];  
    d_world_lidar[i].intensity = d_feats_down_body[i].intensity;
}

__global__ void StateEstimationCudaKernel3(int feats_down_world_size, pointWithVarCuda* d_pv_list, PointCloudXYZICuda* d_feats_down_body,
                                           PointCloudXYZICuda* d_world_lidar, Mat33* d_cross_mat_list, Mat33* d_body_cov_list)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  return;

    d_pv_list[i].point_b[0] = d_feats_down_body[i].x;
    d_pv_list[i].point_b[1] = d_feats_down_body[i].y;
    d_pv_list[i].point_b[2] = d_feats_down_body[i].z;

    d_pv_list[i].point_w[0] = d_world_lidar[i].x;
    d_pv_list[i].point_w[1] = d_world_lidar[i].y;
    d_pv_list[i].point_w[2] = d_world_lidar[i].z;

    double cov[9], point_crossmat[9];
    for (int j=0; j<9; j++){
        cov[j] = d_body_cov_list[i].mat[j];  point_crossmat[j] = d_cross_mat_list[i].mat[j];
    }

    double temp1[9], temp2[9], temp3[9], temp4[9];
    temp1[0] = d_state.rot_end[0]*cov[0] + d_state.rot_end[1]*cov[3] + d_state.rot_end[2]*cov[6];
    temp1[1] = d_state.rot_end[0]*cov[1] + d_state.rot_end[1]*cov[4] + d_state.rot_end[2]*cov[7];
    temp1[2] = d_state.rot_end[0]*cov[2] + d_state.rot_end[1]*cov[5] + d_state.rot_end[2]*cov[8];
    temp1[3] = d_state.rot_end[3]*cov[0] + d_state.rot_end[4]*cov[3] + d_state.rot_end[5]*cov[6];
    temp1[4] = d_state.rot_end[3]*cov[1] + d_state.rot_end[4]*cov[4] + d_state.rot_end[5]*cov[7];
    temp1[5] = d_state.rot_end[3]*cov[2] + d_state.rot_end[4]*cov[5] + d_state.rot_end[5]*cov[8];
    temp1[6] = d_state.rot_end[6]*cov[0] + d_state.rot_end[7]*cov[3] + d_state.rot_end[8]*cov[6];
    temp1[7] = d_state.rot_end[6]*cov[1] + d_state.rot_end[7]*cov[4] + d_state.rot_end[8]*cov[7];
    temp1[8] = d_state.rot_end[6]*cov[2] + d_state.rot_end[7]*cov[5] + d_state.rot_end[8]*cov[8];

    temp2[0] = temp1[0]*d_state.rot_end[0] + temp1[1]*d_state.rot_end[1] + temp1[2]*d_state.rot_end[2];
    temp2[1] = temp1[0]*d_state.rot_end[3] + temp1[1]*d_state.rot_end[4] + temp1[2]*d_state.rot_end[5];
    temp2[2] = temp1[0]*d_state.rot_end[6] + temp1[1]*d_state.rot_end[7] + temp1[2]*d_state.rot_end[8];
    temp2[3] = temp1[3]*d_state.rot_end[0] + temp1[4]*d_state.rot_end[1] + temp1[5]*d_state.rot_end[2];
    temp2[4] = temp1[3]*d_state.rot_end[3] + temp1[4]*d_state.rot_end[4] + temp1[5]*d_state.rot_end[5];
    temp2[5] = temp1[3]*d_state.rot_end[6] + temp1[4]*d_state.rot_end[7] + temp1[5]*d_state.rot_end[8];
    temp2[6] = temp1[6]*d_state.rot_end[0] + temp1[7]*d_state.rot_end[1] + temp1[8]*d_state.rot_end[2];
    temp2[7] = temp1[6]*d_state.rot_end[3] + temp1[7]*d_state.rot_end[4] + temp1[8]*d_state.rot_end[5];
    temp2[8] = temp1[6]*d_state.rot_end[6] + temp1[7]*d_state.rot_end[7] + temp1[8]*d_state.rot_end[8];

    temp3[0] = -point_crossmat[0]*h_d_rot_var[0] - point_crossmat[1]*h_d_rot_var[3] - point_crossmat[2]*h_d_rot_var[6];
    temp3[1] = -point_crossmat[0]*h_d_rot_var[1] - point_crossmat[1]*h_d_rot_var[4] - point_crossmat[2]*h_d_rot_var[7];
    temp3[2] = -point_crossmat[0]*h_d_rot_var[2] - point_crossmat[1]*h_d_rot_var[5] - point_crossmat[2]*h_d_rot_var[8];
    temp3[3] = -point_crossmat[3]*h_d_rot_var[0] - point_crossmat[4]*h_d_rot_var[3] - point_crossmat[5]*h_d_rot_var[6];
    temp3[4] = -point_crossmat[3]*h_d_rot_var[1] - point_crossmat[4]*h_d_rot_var[4] - point_crossmat[5]*h_d_rot_var[7];
    temp3[5] = -point_crossmat[3]*h_d_rot_var[2] - point_crossmat[4]*h_d_rot_var[5] - point_crossmat[5]*h_d_rot_var[8];
    temp3[6] = -point_crossmat[6]*h_d_rot_var[0] - point_crossmat[7]*h_d_rot_var[3] - point_crossmat[8]*h_d_rot_var[6];
    temp3[7] = -point_crossmat[6]*h_d_rot_var[1] - point_crossmat[7]*h_d_rot_var[4] - point_crossmat[8]*h_d_rot_var[7];
    temp3[8] = -point_crossmat[6]*h_d_rot_var[2] - point_crossmat[7]*h_d_rot_var[5] - point_crossmat[8]*h_d_rot_var[8];

    temp4[0] = -temp3[0]*point_crossmat[0] - temp3[1]*point_crossmat[1] - temp3[2]*point_crossmat[2];
    temp4[1] = -temp3[0]*point_crossmat[3] - temp3[1]*point_crossmat[4] - temp3[2]*point_crossmat[5];
    temp4[2] = -temp3[0]*point_crossmat[6] - temp3[1]*point_crossmat[7] - temp3[2]*point_crossmat[8];
    temp4[3] = -temp3[3]*point_crossmat[0] - temp3[4]*point_crossmat[1] - temp3[5]*point_crossmat[2];
    temp4[4] = -temp3[3]*point_crossmat[3] - temp3[4]*point_crossmat[4] - temp3[5]*point_crossmat[5];
    temp4[5] = -temp3[3]*point_crossmat[6] - temp3[4]*point_crossmat[7] - temp3[5]*point_crossmat[8];
    temp4[6] = -temp3[6]*point_crossmat[0] - temp3[7]*point_crossmat[1] - temp3[8]*point_crossmat[2];
    temp4[7] = -temp3[6]*point_crossmat[3] - temp3[7]*point_crossmat[4] - temp3[8]*point_crossmat[5];
    temp4[8] = -temp3[6]*point_crossmat[6] - temp3[7]*point_crossmat[7] - temp3[8]*point_crossmat[8];

    for (int j=0; j<9; j++){
        d_pv_list[i].var[j] = temp2[j] + temp4[j] + h_d_t_var[j];
        d_pv_list[i].body_var[j] = d_body_cov_list[i].mat[j];
    }
}

__global__ void StateEstimationCudaKernel4(int effect_feat_num, PointToPlaneCuda* d_ptpl_list, double* d_Hsub, double* d_Hsub_T_R_inv, 
                                           double* d_R_inv, double* d_meas_vec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= effect_feat_num)  return;

    PointToPlaneCuda &ptpl = d_ptpl_list[i];
    double point_this[3] = {ptpl.point_b_[0], ptpl.point_b_[1], ptpl.point_b_[2]};
    double temp1[3];
    temp1[0] = d_ParametersCuda.extR[0]*point_this[0] + d_ParametersCuda.extR[1]*point_this[1] + 
               d_ParametersCuda.extR[2]*point_this[2] + d_ParametersCuda.extT[0];
    temp1[1] = d_ParametersCuda.extR[3]*point_this[0] + d_ParametersCuda.extR[4]*point_this[1] + 
               d_ParametersCuda.extR[5]*point_this[2] + d_ParametersCuda.extT[1];
    temp1[2] = d_ParametersCuda.extR[6]*point_this[0] + d_ParametersCuda.extR[7]*point_this[1] + 
               d_ParametersCuda.extR[8]*point_this[2] + d_ParametersCuda.extT[2];
    point_this[0] = temp1[0];  point_this[1] = temp1[1];  point_this[2] = temp1[2];
    double point_crossmat[9] = {0.0, -point_this[2], point_this[1], point_this[2], 0.0, -point_this[0], -point_this[1], point_this[0], 0.0};

    double point_world[3];
    point_world[0] = d_state_propagate.rot_end[0]*point_this[0] + d_state_propagate.rot_end[1]*point_this[1] + 
                     d_state_propagate.rot_end[2]*point_this[2] + d_state_propagate.pos_end[0];
    point_world[1] = d_state_propagate.rot_end[3]*point_this[0] + d_state_propagate.rot_end[4]*point_this[1] + 
                     d_state_propagate.rot_end[5]*point_this[2] + d_state_propagate.pos_end[1];
    point_world[2] = d_state_propagate.rot_end[6]*point_this[0] + d_state_propagate.rot_end[7]*point_this[1] + 
                     d_state_propagate.rot_end[8]*point_this[2] + d_state_propagate.pos_end[2];

    double J_nq[6];
    J_nq[0] = point_world[0] - ptpl.center_[0];
    J_nq[1] = point_world[1] - ptpl.center_[1];
    J_nq[2] = point_world[2] - ptpl.center_[2];
    J_nq[3] = -ptpl.normal_[0];
    J_nq[4] = -ptpl.normal_[1];
    J_nq[5] = -ptpl.normal_[2];

    double var[9], temp2[9], temp3[9];
    temp2[0] = d_state_propagate.rot_end[0]*d_ParametersCuda.extR[0] + d_state_propagate.rot_end[1]*d_ParametersCuda.extR[3] + 
               d_state_propagate.rot_end[2]*d_ParametersCuda.extR[6];
    temp2[1] = d_state_propagate.rot_end[0]*d_ParametersCuda.extR[1] + d_state_propagate.rot_end[1]*d_ParametersCuda.extR[4] + 
               d_state_propagate.rot_end[2]*d_ParametersCuda.extR[7];
    temp2[2] = d_state_propagate.rot_end[0]*d_ParametersCuda.extR[2] + d_state_propagate.rot_end[1]*d_ParametersCuda.extR[5] + 
               d_state_propagate.rot_end[2]*d_ParametersCuda.extR[8];
    temp2[3] = d_state_propagate.rot_end[3]*d_ParametersCuda.extR[0] + d_state_propagate.rot_end[4]*d_ParametersCuda.extR[3] + 
               d_state_propagate.rot_end[5]*d_ParametersCuda.extR[6];
    temp2[4] = d_state_propagate.rot_end[3]*d_ParametersCuda.extR[1] + d_state_propagate.rot_end[4]*d_ParametersCuda.extR[4] + 
               d_state_propagate.rot_end[5]*d_ParametersCuda.extR[7];
    temp2[5] = d_state_propagate.rot_end[3]*d_ParametersCuda.extR[2] + d_state_propagate.rot_end[4]*d_ParametersCuda.extR[5] + 
               d_state_propagate.rot_end[5]*d_ParametersCuda.extR[8];
    temp2[6] = d_state_propagate.rot_end[6]*d_ParametersCuda.extR[0] + d_state_propagate.rot_end[7]*d_ParametersCuda.extR[3] + 
               d_state_propagate.rot_end[8]*d_ParametersCuda.extR[6];
    temp2[7] = d_state_propagate.rot_end[6]*d_ParametersCuda.extR[1] + d_state_propagate.rot_end[7]*d_ParametersCuda.extR[4] + 
               d_state_propagate.rot_end[8]*d_ParametersCuda.extR[7];
    temp2[8] = d_state_propagate.rot_end[6]*d_ParametersCuda.extR[2] + d_state_propagate.rot_end[7]*d_ParametersCuda.extR[5] + 
               d_state_propagate.rot_end[8]*d_ParametersCuda.extR[8];

    temp3[0] = temp2[0]*ptpl.body_cov_[0] + temp2[1]*ptpl.body_cov_[3] + temp2[2]*ptpl.body_cov_[6];
    temp3[1] = temp2[0]*ptpl.body_cov_[1] + temp2[1]*ptpl.body_cov_[4] + temp2[2]*ptpl.body_cov_[7];
    temp3[2] = temp2[0]*ptpl.body_cov_[2] + temp2[1]*ptpl.body_cov_[5] + temp2[2]*ptpl.body_cov_[8];
    temp3[3] = temp2[3]*ptpl.body_cov_[0] + temp2[4]*ptpl.body_cov_[3] + temp2[5]*ptpl.body_cov_[6];
    temp3[4] = temp2[3]*ptpl.body_cov_[1] + temp2[4]*ptpl.body_cov_[4] + temp2[5]*ptpl.body_cov_[7];
    temp3[5] = temp2[3]*ptpl.body_cov_[2] + temp2[4]*ptpl.body_cov_[5] + temp2[5]*ptpl.body_cov_[8];
    temp3[6] = temp2[6]*ptpl.body_cov_[0] + temp2[7]*ptpl.body_cov_[3] + temp2[8]*ptpl.body_cov_[6];
    temp3[7] = temp2[6]*ptpl.body_cov_[1] + temp2[7]*ptpl.body_cov_[4] + temp2[8]*ptpl.body_cov_[7];
    temp3[8] = temp2[6]*ptpl.body_cov_[2] + temp2[7]*ptpl.body_cov_[5] + temp2[8]*ptpl.body_cov_[8];

    var[0] = temp3[0]*temp2[0] + temp3[1]*temp2[1] + temp3[2]*temp2[2];
    var[1] = temp3[0]*temp2[3] + temp3[1]*temp2[4] + temp3[2]*temp2[5];
    var[2] = temp3[0]*temp2[6] + temp3[1]*temp2[7] + temp3[2]*temp2[8];
    var[3] = temp3[3]*temp2[0] + temp3[4]*temp2[1] + temp3[5]*temp2[2];
    var[4] = temp3[3]*temp2[3] + temp3[4]*temp2[4] + temp3[5]*temp2[5];
    var[5] = temp3[3]*temp2[6] + temp3[4]*temp2[7] + temp3[5]*temp2[8];
    var[6] = temp3[6]*temp2[0] + temp3[7]*temp2[1] + temp3[8]*temp2[2];
    var[7] = temp3[6]*temp2[3] + temp3[7]*temp2[4] + temp3[8]*temp2[5];
    var[8] = temp3[6]*temp2[6] + temp3[7]*temp2[7] + temp3[8]*temp2[8];

    double temp4[6];
    temp4[0] = J_nq[0]*ptpl.plane_var_[0]  + J_nq[1]*ptpl.plane_var_[6]  + J_nq[2]*ptpl.plane_var_[12] +
               J_nq[3]*ptpl.plane_var_[18] + J_nq[4]*ptpl.plane_var_[24] + J_nq[5]*ptpl.plane_var_[30];
    temp4[1] = J_nq[0]*ptpl.plane_var_[1]  + J_nq[1]*ptpl.plane_var_[7]  + J_nq[2]*ptpl.plane_var_[13] +
               J_nq[3]*ptpl.plane_var_[19] + J_nq[4]*ptpl.plane_var_[25] + J_nq[5]*ptpl.plane_var_[31];
    temp4[2] = J_nq[0]*ptpl.plane_var_[2]  + J_nq[1]*ptpl.plane_var_[8]  + J_nq[2]*ptpl.plane_var_[14] +
               J_nq[3]*ptpl.plane_var_[20] + J_nq[4]*ptpl.plane_var_[26] + J_nq[5]*ptpl.plane_var_[32];
    temp4[3] = J_nq[0]*ptpl.plane_var_[3]  + J_nq[1]*ptpl.plane_var_[9]  + J_nq[2]*ptpl.plane_var_[15] +
               J_nq[3]*ptpl.plane_var_[21] + J_nq[4]*ptpl.plane_var_[27] + J_nq[5]*ptpl.plane_var_[33];
    temp4[4] = J_nq[0]*ptpl.plane_var_[4]  + J_nq[1]*ptpl.plane_var_[10] + J_nq[2]*ptpl.plane_var_[16] +
               J_nq[3]*ptpl.plane_var_[22] + J_nq[4]*ptpl.plane_var_[28] + J_nq[5]*ptpl.plane_var_[34];
    temp4[5] = J_nq[0]*ptpl.plane_var_[5]  + J_nq[1]*ptpl.plane_var_[11] + J_nq[2]*ptpl.plane_var_[17] +
               J_nq[3]*ptpl.plane_var_[23] + J_nq[4]*ptpl.plane_var_[29] + J_nq[5]*ptpl.plane_var_[35];

    double sigma_l = temp4[0]*J_nq[0] + temp4[1]*J_nq[1] + temp4[2]*J_nq[2] +
                     temp4[3]*J_nq[3] + temp4[4]*J_nq[4] + temp4[5]*J_nq[5];
                    
    double temp5[3], temp6;
    temp5[0] = ptpl.normal_[0]*var[0] + ptpl.normal_[1]*var[3] + ptpl.normal_[2]*var[6];
    temp5[1] = ptpl.normal_[0]*var[1] + ptpl.normal_[1]*var[4] + ptpl.normal_[2]*var[7];
    temp5[2] = ptpl.normal_[0]*var[2] + ptpl.normal_[1]*var[5] + ptpl.normal_[2]*var[8];

    temp6 = temp5[0]*ptpl.normal_[0] + temp5[1]*ptpl.normal_[1] + temp5[2]*ptpl.normal_[2];
    d_R_inv[i] = 1.0 / (0.001 + sigma_l + temp6);

    double A[3], temp7[9];

    temp7[0] = point_crossmat[0]*d_state_propagate.rot_end[0] + point_crossmat[1]*d_state_propagate.rot_end[1] + 
               point_crossmat[2]*d_state_propagate.rot_end[2];
    temp7[1] = point_crossmat[0]*d_state_propagate.rot_end[3] + point_crossmat[1]*d_state_propagate.rot_end[4] + 
               point_crossmat[2]*d_state_propagate.rot_end[5];
    temp7[2] = point_crossmat[0]*d_state_propagate.rot_end[6] + point_crossmat[1]*d_state_propagate.rot_end[7] + 
               point_crossmat[2]*d_state_propagate.rot_end[8];
    temp7[3] = point_crossmat[3]*d_state_propagate.rot_end[0] + point_crossmat[4]*d_state_propagate.rot_end[1] + 
               point_crossmat[5]*d_state_propagate.rot_end[2];
    temp7[4] = point_crossmat[3]*d_state_propagate.rot_end[3] + point_crossmat[4]*d_state_propagate.rot_end[4] + 
               point_crossmat[5]*d_state_propagate.rot_end[5];
    temp7[5] = point_crossmat[3]*d_state_propagate.rot_end[6] + point_crossmat[4]*d_state_propagate.rot_end[7] + 
               point_crossmat[5]*d_state_propagate.rot_end[8];
    temp7[6] = point_crossmat[6]*d_state_propagate.rot_end[0] + point_crossmat[7]*d_state_propagate.rot_end[1] + 
               point_crossmat[8]*d_state_propagate.rot_end[2];
    temp7[7] = point_crossmat[6]*d_state_propagate.rot_end[3] + point_crossmat[7]*d_state_propagate.rot_end[4] + 
               point_crossmat[8]*d_state_propagate.rot_end[5];
    temp7[8] = point_crossmat[6]*d_state_propagate.rot_end[6] + point_crossmat[7]*d_state_propagate.rot_end[7] + 
               point_crossmat[8]*d_state_propagate.rot_end[8];

    A[0] = temp7[0]*ptpl.normal_[0] + temp7[1]*ptpl.normal_[1] + temp7[2]*ptpl.normal_[2];
    A[1] = temp7[3]*ptpl.normal_[0] + temp7[4]*ptpl.normal_[1] + temp7[5]*ptpl.normal_[2];
    A[2] = temp7[6]*ptpl.normal_[0] + temp7[7]*ptpl.normal_[1] + temp7[8]*ptpl.normal_[2];

    d_Hsub[6*i] = A[0];  d_Hsub[6*i+1] = A[1];  d_Hsub[6*i+2] = A[2];
    d_Hsub[6*i+3] = ptpl.normal_[0];  d_Hsub[6*i+4] = ptpl.normal_[1];  d_Hsub[6*i+5] = ptpl.normal_[2];

    d_Hsub_T_R_inv[effect_feat_num*0+i] = A[0] * d_R_inv[i];
    d_Hsub_T_R_inv[effect_feat_num*1+i] = A[1] * d_R_inv[i];
    d_Hsub_T_R_inv[effect_feat_num*2+i] = A[2] * d_R_inv[i];
    d_Hsub_T_R_inv[effect_feat_num*3+i] = ptpl.normal_[0] * d_R_inv[i];
    d_Hsub_T_R_inv[effect_feat_num*4+i] = ptpl.normal_[1] * d_R_inv[i];
    d_Hsub_T_R_inv[effect_feat_num*5+i] = ptpl.normal_[2] * d_R_inv[i];

    d_meas_vec[i] = -ptpl.dis_to_plane_;
}

__global__ void BuildVoxelMapCudaKernel1()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_CAPACITY)  return;

    free_stack_root[i] = i;
    free_stack_leaf[i] = i;
}

template <typename FindRefRoot, typename InsertRefRoot>
__global__ void BuildVoxelMapCudaKernel2(int feats_down_world_size, PointCloudXYZICuda* d_feats_down_world,
                                        PointCloudXYZICuda* d_feats_down_body, pointWithVarCuda* d_input_points,
                                        VOXEL_LOCATION_CUDA* d_root_voxel_location_cuda, RootVoxelCuda* d_root_voxel_cuda,
                                        double dept_err, double beam_err, float voxel_size, int layer_init_num_0, 
                                        int layer_init_num_1, int layer_init_num_2, int layer_init_num_3, int layer_init_num_4,
                                        float planner_threshold,
                                        FindRefRoot find_ref_root, InsertRefRoot insert_ref_root)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  {return;}
    
    d_input_points[i].point_w[0] = d_feats_down_world[i].x;
    d_input_points[i].point_w[1] = d_feats_down_world[i].y;
    d_input_points[i].point_w[2] = d_feats_down_world[i].z;

    double point_this[3] = {d_feats_down_body[i].x, d_feats_down_body[i].y, d_feats_down_body[i].z};
    double var[9];
    calcBodyCovCuda(point_this, dept_err, beam_err, var);

    double point_crossmat[9] = {0.0, -point_this[2], point_this[1], point_this[2], 0.0, -point_this[0], -point_this[1], point_this[0], 0.0};

    {
    double temp1[9], temp2[9], temp3[9], temp4[9], temp5[9];
    temp1[0] = d_state.rot_end[0]*d_ParametersCuda.extR[0] + d_state.rot_end[1]*d_ParametersCuda.extR[3] + 
               d_state.rot_end[2]*d_ParametersCuda.extR[6];
    temp1[1] = d_state.rot_end[0]*d_ParametersCuda.extR[1] + d_state.rot_end[1]*d_ParametersCuda.extR[4] + 
               d_state.rot_end[2]*d_ParametersCuda.extR[7];
    temp1[2] = d_state.rot_end[0]*d_ParametersCuda.extR[2] + d_state.rot_end[1]*d_ParametersCuda.extR[5] + 
               d_state.rot_end[2]*d_ParametersCuda.extR[8];
    temp1[3] = d_state.rot_end[3]*d_ParametersCuda.extR[0] + d_state.rot_end[4]*d_ParametersCuda.extR[3] + 
               d_state.rot_end[5]*d_ParametersCuda.extR[6];
    temp1[4] = d_state.rot_end[3]*d_ParametersCuda.extR[1] + d_state.rot_end[4]*d_ParametersCuda.extR[4] + 
               d_state.rot_end[5]*d_ParametersCuda.extR[7];
    temp1[5] = d_state.rot_end[3]*d_ParametersCuda.extR[2] + d_state.rot_end[4]*d_ParametersCuda.extR[5] + 
               d_state.rot_end[5]*d_ParametersCuda.extR[8];
    temp1[6] = d_state.rot_end[6]*d_ParametersCuda.extR[0] + d_state.rot_end[7]*d_ParametersCuda.extR[3] + 
               d_state.rot_end[8]*d_ParametersCuda.extR[6];
    temp1[7] = d_state.rot_end[6]*d_ParametersCuda.extR[1] + d_state.rot_end[7]*d_ParametersCuda.extR[4] + 
               d_state.rot_end[8]*d_ParametersCuda.extR[7];
    temp1[8] = d_state.rot_end[6]*d_ParametersCuda.extR[2] + d_state.rot_end[7]*d_ParametersCuda.extR[5] + 
               d_state.rot_end[8]*d_ParametersCuda.extR[8];

    temp2[0] = temp1[0]*var[0] + temp1[1]*var[3] + temp1[2]*var[6];
    temp2[1] = temp1[0]*var[1] + temp1[1]*var[4] + temp1[2]*var[7];
    temp2[2] = temp1[0]*var[2] + temp1[1]*var[5] + temp1[2]*var[8];
    temp2[3] = temp1[3]*var[0] + temp1[4]*var[3] + temp1[5]*var[6];
    temp2[4] = temp1[3]*var[1] + temp1[4]*var[4] + temp1[5]*var[7];
    temp2[5] = temp1[3]*var[2] + temp1[4]*var[5] + temp1[5]*var[8];
    temp2[6] = temp1[6]*var[0] + temp1[7]*var[3] + temp1[8]*var[6];
    temp2[7] = temp1[6]*var[1] + temp1[7]*var[4] + temp1[8]*var[7];
    temp2[8] = temp1[6]*var[2] + temp1[7]*var[5] + temp1[8]*var[8];

    temp3[0] = temp2[0]*temp1[0] + temp2[1]*temp1[1] + temp2[2]*temp1[2];
    temp3[1] = temp2[0]*temp1[3] + temp2[1]*temp1[4] + temp2[2]*temp1[5];
    temp3[2] = temp2[0]*temp1[6] + temp2[1]*temp1[7] + temp2[2]*temp1[8];
    temp3[3] = temp2[3]*temp1[0] + temp2[4]*temp1[1] + temp2[5]*temp1[2];
    temp3[4] = temp2[3]*temp1[3] + temp2[4]*temp1[4] + temp2[5]*temp1[5];
    temp3[5] = temp2[3]*temp1[6] + temp2[4]*temp1[7] + temp2[5]*temp1[8];
    temp3[6] = temp2[6]*temp1[0] + temp2[7]*temp1[1] + temp2[8]*temp1[2];
    temp3[7] = temp2[6]*temp1[3] + temp2[7]*temp1[4] + temp2[8]*temp1[5];
    temp3[8] = temp2[6]*temp1[6] + temp2[7]*temp1[7] + temp2[8]*temp1[8];

    temp4[0] = -point_crossmat[0]*d_state.cov[0] - point_crossmat[1]*d_state.cov[19] - point_crossmat[2]*d_state.cov[38];
    temp4[1] = -point_crossmat[0]*d_state.cov[1] - point_crossmat[1]*d_state.cov[20] - point_crossmat[2]*d_state.cov[39];
    temp4[2] = -point_crossmat[0]*d_state.cov[2] - point_crossmat[1]*d_state.cov[21] - point_crossmat[2]*d_state.cov[40];
    temp4[3] = -point_crossmat[3]*d_state.cov[0] - point_crossmat[4]*d_state.cov[19] - point_crossmat[5]*d_state.cov[38];
    temp4[4] = -point_crossmat[3]*d_state.cov[1] - point_crossmat[4]*d_state.cov[20] - point_crossmat[5]*d_state.cov[39];
    temp4[5] = -point_crossmat[3]*d_state.cov[2] - point_crossmat[4]*d_state.cov[21] - point_crossmat[5]*d_state.cov[40];
    temp4[6] = -point_crossmat[6]*d_state.cov[0] - point_crossmat[7]*d_state.cov[19] - point_crossmat[8]*d_state.cov[38];
    temp4[7] = -point_crossmat[6]*d_state.cov[1] - point_crossmat[7]*d_state.cov[20] - point_crossmat[8]*d_state.cov[39];
    temp4[8] = -point_crossmat[6]*d_state.cov[2] - point_crossmat[7]*d_state.cov[21] - point_crossmat[8]*d_state.cov[40];

    temp5[0] = -temp4[0]*point_crossmat[0] - temp4[1]*point_crossmat[1] - temp4[2]*point_crossmat[2];
    temp5[1] = -temp4[0]*point_crossmat[3] - temp4[1]*point_crossmat[4] - temp4[2]*point_crossmat[5];
    temp5[2] = -temp4[0]*point_crossmat[6] - temp4[1]*point_crossmat[7] - temp4[2]*point_crossmat[8];
    temp5[3] = -temp4[3]*point_crossmat[0] - temp4[4]*point_crossmat[1] - temp4[5]*point_crossmat[2];
    temp5[4] = -temp4[3]*point_crossmat[3] - temp4[4]*point_crossmat[4] - temp4[5]*point_crossmat[5];
    temp5[5] = -temp4[3]*point_crossmat[6] - temp4[4]*point_crossmat[7] - temp4[5]*point_crossmat[8];
    temp5[6] = -temp4[6]*point_crossmat[0] - temp4[7]*point_crossmat[1] - temp4[8]*point_crossmat[2];
    temp5[7] = -temp4[6]*point_crossmat[3] - temp4[7]*point_crossmat[4] - temp4[8]*point_crossmat[5];
    temp5[8] = -temp4[6]*point_crossmat[6] - temp4[7]*point_crossmat[7] - temp4[8]*point_crossmat[8];

    var[0] = temp3[0] + temp5[0] + d_state.cov[60];
    var[1] = temp3[1] + temp5[1] + d_state.cov[61];
    var[2] = temp3[2] + temp5[2] + d_state.cov[62];
    var[3] = temp3[3] + temp5[3] + d_state.cov[79];
    var[4] = temp3[4] + temp5[4] + d_state.cov[80];
    var[5] = temp3[5] + temp5[5] + d_state.cov[81];
    var[6] = temp3[6] + temp5[6] + d_state.cov[98];
    var[7] = temp3[7] + temp5[7] + d_state.cov[99];
    var[8] = temp3[8] + temp5[8] + d_state.cov[100];

    for (int j=0; j<9; j++)  {d_input_points[i].var[j] = var[j];}
    }

    const pointWithVarCuda p_v = d_input_points[i];

    float loc_xyz[3];
    for (int j=0; j<3; j++)
    {
        loc_xyz[j] = p_v.point_w[j] / voxel_size;
        if (loc_xyz[j] < 0)  {loc_xyz[j] -= 1.0;}
    }
    VOXEL_LOCATION_CUDA position;
    position.x = (int64_t)loc_xyz[0]; position.y = (int64_t)loc_xyz[1]; position.z = (int64_t)loc_xyz[2];

    auto iter = find_ref_root.find(&position);
    if (iter != find_ref_root.end())
    {
        int point_index = atomicAdd(&(iter->second->new_points_), 1);
        if (point_index >= 100)
        {
            atomicExch(&(iter->second->new_points_), 100);
            atomicExch(&(iter->second->update_enable_), 0);
            return;
        }
        iter->second->temp_points_[point_index] = p_v;
    }
    else
    {
        int current_free_bottom_root = atomicAdd(&free_bottom_root, 1);
        int current_root_voxel_index = free_stack_root[current_free_bottom_root];
        if (current_root_voxel_index >= TOTAL_CAPACITY)  {return;}
        d_root_voxel_location_cuda[current_root_voxel_index] = position;
        d_root_voxel_cuda[current_root_voxel_index].voxel_center_[0] = (0.5 + position.x) * voxel_size;
        d_root_voxel_cuda[current_root_voxel_index].voxel_center_[1] = (0.5 + position.y) * voxel_size;
        d_root_voxel_cuda[current_root_voxel_index].voxel_center_[2] = (0.5 + position.z) * voxel_size;
        
        for (int dx=0; dx<2; ++dx){
            for (int dy=0; dy<2; ++dy){
                for (int dz=0; dz<2; ++dz){
                    d_root_voxel_cuda[current_root_voxel_index].leaf_voxel_[dz+dy*2+dx*4].x = position.x * 2 + dx;
                    d_root_voxel_cuda[current_root_voxel_index].leaf_voxel_[dz+dy*2+dx*4].y = position.y * 2 + dy;
                    d_root_voxel_cuda[current_root_voxel_index].leaf_voxel_[dz+dy*2+dx*4].z = position.z * 2 + dz;
                }
            }
        }

        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[0] = layer_init_num_0;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[1] = layer_init_num_1;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[2] = layer_init_num_2;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[3] = layer_init_num_3;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[4] = layer_init_num_4;
        d_root_voxel_cuda[current_root_voxel_index].points_size_threshold_ = layer_init_num_0;
        d_root_voxel_cuda[current_root_voxel_index].update_size_threshold_ = 5;
        d_root_voxel_cuda[current_root_voxel_index].new_points_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].quater_length_ = voxel_size / 4;
        d_root_voxel_cuda[current_root_voxel_index].planer_threshold_ = planner_threshold;
        d_root_voxel_cuda[current_root_voxel_index].update_enable_ = 1;
        d_root_voxel_cuda[current_root_voxel_index].leaf_enable_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].init_octo_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].is_valid_ = 0;
        
        d_root_voxel_cuda[current_root_voxel_index].plane_.radius_ = 0.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.min_eigen_value_ = 1.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.mid_eigen_value_ = 1.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.max_eigen_value_ = 1.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.d_ = 0.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.points_size_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].plane_.id_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].plane_.is_plane_ = false;
        d_root_voxel_cuda[current_root_voxel_index].plane_.is_init_ = false;
        d_root_voxel_cuda[current_root_voxel_index].plane_.is_update_ = false;

        int point_index = atomicAdd(&(d_root_voxel_cuda[current_root_voxel_index].new_points_), 1);
        d_root_voxel_cuda[current_root_voxel_index].temp_points_[point_index] = p_v;
        bool is_inserted = insert_ref_root.insert(cuco::pair{&d_root_voxel_location_cuda[current_root_voxel_index], 
                                                             &d_root_voxel_cuda[current_root_voxel_index]});
        if (is_inserted)  d_root_voxel_cuda[current_root_voxel_index].is_valid_ = 1;
        else
        {
            d_root_voxel_cuda[current_root_voxel_index].is_valid_ = 0;
            auto iter1 = find_ref_root.find(&position);
            if (iter1 != find_ref_root.end())
            {
                int point_index = atomicAdd(&(iter1->second->new_points_), 1);
                if (point_index >= 100)
                {
                    atomicExch(&(iter1->second->new_points_), 100);
                    atomicExch(&(iter1->second->update_enable_), 0);
                    return;
                }
                iter1->second->temp_points_[point_index] = p_v;
            }
        }
    }
}

template <typename FindRefRoot, typename FindRefLeaf>
__global__ void BuildResidualListOMPCudaKernel1(int feats_down_world_size, pointWithVarCuda* d_pv_list, PointToPlaneCuda* d_all_ptpl_list, 
                                           bool* d_useful_ptpl, FindRefRoot find_ref_root, FindRefLeaf find_ref_leaf, 
                                           double voxel_size, double sigma_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  return;

    d_useful_ptpl[i] = false;
    
    pointWithVarCuda &pv = d_pv_list[i];
    float loc_xyz[3];
    for (int j=0; j<3; j++)
    {
        loc_xyz[j] = pv.point_w[j] / voxel_size;
        if (loc_xyz[j] < 0)  loc_xyz[j] -= 1.0;
    }
    
    VOXEL_LOCATION_CUDA position;
    position.x = (int64_t)loc_xyz[0];  position.y = (int64_t)loc_xyz[1];  position.z = (int64_t)loc_xyz[2];

    auto iter = find_ref_root.find(&position);
    if (iter != find_ref_root.end())
    {
        RootVoxelCuda* root_value = iter->second;
        PointToPlaneCuda &single_ptpl = d_all_ptpl_list[i];
        bool is_success = false;
        double prob = 0;
        build_single_residual_root_cuda(pv, root_value, prob, is_success, single_ptpl, sigma_num, find_ref_leaf, i);

        if (!is_success)
        {
            VOXEL_LOCATION_CUDA near_position = position;
            if (loc_xyz[0] > (root_value->voxel_center_[0] + root_value->quater_length_))  near_position.x = near_position.x + 1;
            else if (loc_xyz[0] < (root_value->voxel_center_[0] - root_value->quater_length_))  near_position.x = near_position.x - 1;
            if (loc_xyz[1] > (root_value->voxel_center_[1] + root_value->quater_length_))  near_position.y = near_position.y + 1;
            else if (loc_xyz[1] < (root_value->voxel_center_[1] - root_value->quater_length_))  near_position.y = near_position.y - 1;
            if (loc_xyz[2] > (root_value->voxel_center_[2] + root_value->quater_length_))  near_position.z = near_position.z + 1;
            else if (loc_xyz[2] < (root_value->voxel_center_[2] - root_value->quater_length_))  near_position.z = near_position.z - 1;
            auto iter_near = find_ref_root.find(&near_position);
            if (iter_near != find_ref_root.end())
            {
                RootVoxelCuda* root_value_near = iter_near->second;
                build_single_residual_root_cuda(pv, root_value_near, prob, is_success, single_ptpl, sigma_num, find_ref_leaf, i);
            }
        }
        if (is_success)  d_useful_ptpl[i] = true;
        else  d_useful_ptpl[i] = false;
    }
}

template <typename FindRefRoot, typename InsertRefRoot>
__global__ void UpdateVoxelMapCudaKernel1(int feats_down_world_size, pointWithVarCuda* d_pv_list, VOXEL_LOCATION_CUDA* d_root_voxel_location_cuda,
                                RootVoxelCuda* d_root_voxel_cuda, float voxel_size, int layer_init_num_0, int layer_init_num_1, 
                                int layer_init_num_2, int layer_init_num_3, int layer_init_num_4, float planner_threshold,
                                FindRefRoot find_ref_root, InsertRefRoot insert_ref_root)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  return;

    const pointWithVarCuda p_v = d_pv_list[i];
    float loc_xyz[3];
    for (int j=0; j<3; j++)
    {
        loc_xyz[j] = p_v.point_w[j] / voxel_size;
        if (loc_xyz[j] < 0)  {loc_xyz[j] -= 1.0;}
    }
    VOXEL_LOCATION_CUDA position;
    position.x = (int64_t)loc_xyz[0]; position.y = (int64_t)loc_xyz[1]; position.z = (int64_t)loc_xyz[2];
    
    auto iter = find_ref_root.find(&position);
    if (iter != find_ref_root.end())
    {
        int point_index = atomicAdd(&(iter->second->new_points_), 1);
        if (point_index >= 100)
        {
            atomicExch(&(iter->second->new_points_), 100);
            atomicExch(&(iter->second->update_enable_), 0);
            return;
        }
        iter->second->temp_points_[point_index] = p_v;
    }
    else
    {
        int current_free_bottom_root = atomicAdd(&free_bottom_root, 1);
        int current_root_voxel_index = free_stack_root[current_free_bottom_root];
        if (current_root_voxel_index >= TOTAL_CAPACITY)  {return;}
        d_root_voxel_location_cuda[current_root_voxel_index] = position;
        d_root_voxel_cuda[current_root_voxel_index].voxel_center_[0] = (0.5 + position.x) * voxel_size;
        d_root_voxel_cuda[current_root_voxel_index].voxel_center_[1] = (0.5 + position.y) * voxel_size;
        d_root_voxel_cuda[current_root_voxel_index].voxel_center_[2] = (0.5 + position.z) * voxel_size;
        
        for (int dx=0; dx<2; ++dx){
            for (int dy=0; dy<2; ++dy){
                for (int dz=0; dz<2; ++dz){
                    d_root_voxel_cuda[current_root_voxel_index].leaf_voxel_[dz+dy*2+dx*4].x = position.x * 2 + dx;
                    d_root_voxel_cuda[current_root_voxel_index].leaf_voxel_[dz+dy*2+dx*4].y = position.y * 2 + dy;
                    d_root_voxel_cuda[current_root_voxel_index].leaf_voxel_[dz+dy*2+dx*4].z = position.z * 2 + dz;
                }
            }
        }

        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[0] = layer_init_num_0;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[1] = layer_init_num_1;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[2] = layer_init_num_2;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[3] = layer_init_num_3;
        d_root_voxel_cuda[current_root_voxel_index].layer_init_num_[4] = layer_init_num_4;
        d_root_voxel_cuda[current_root_voxel_index].points_size_threshold_ = layer_init_num_0;
        d_root_voxel_cuda[current_root_voxel_index].update_size_threshold_ = 5;
        d_root_voxel_cuda[current_root_voxel_index].new_points_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].quater_length_ = voxel_size / 4;
        d_root_voxel_cuda[current_root_voxel_index].planer_threshold_ = planner_threshold;
        d_root_voxel_cuda[current_root_voxel_index].update_enable_ = 1;
        d_root_voxel_cuda[current_root_voxel_index].leaf_enable_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].init_octo_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].is_valid_ = 0;
        
        d_root_voxel_cuda[current_root_voxel_index].plane_.radius_ = 0.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.min_eigen_value_ = 1.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.mid_eigen_value_ = 1.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.max_eigen_value_ = 1.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.d_ = 0.0f;
        d_root_voxel_cuda[current_root_voxel_index].plane_.points_size_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].plane_.id_ = 0;
        d_root_voxel_cuda[current_root_voxel_index].plane_.is_plane_ = false;
        d_root_voxel_cuda[current_root_voxel_index].plane_.is_init_ = false;
        d_root_voxel_cuda[current_root_voxel_index].plane_.is_update_ = false;

        int point_index = atomicAdd(&(d_root_voxel_cuda[current_root_voxel_index].new_points_), 1);
        d_root_voxel_cuda[current_root_voxel_index].temp_points_[point_index] = p_v;
        bool is_inserted = insert_ref_root.insert(cuco::pair{&d_root_voxel_location_cuda[current_root_voxel_index], 
                                                             &d_root_voxel_cuda[current_root_voxel_index]});
        if (is_inserted)  d_root_voxel_cuda[current_root_voxel_index].is_valid_ = 1;
        else
        {
            d_root_voxel_cuda[current_root_voxel_index].is_valid_ = 0;
            auto iter1 = find_ref_root.find(&position);
            if (iter1 != find_ref_root.end())
            {
                int point_index = atomicAdd(&(iter1->second->new_points_), 1);
                if (point_index >= 100)
                {
                    atomicExch(&(iter1->second->new_points_), 100);
                    atomicExch(&(iter1->second->update_enable_), 0);
                    return;
                }
                iter1->second->temp_points_[point_index] = p_v;
            }
        }
    }
}

__global__ void UpdateVoxelMapCudaKernel2(int max_index_root, RootVoxelCuda* d_root_voxel_cuda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_index_root)  return;

    if (d_root_voxel_cuda[i].is_valid_ == 0)
    {
        int current_root_bottom = atomicAdd(&free_bottom_root_count, 1);
        free_stack_root[current_root_bottom] = i;
    }
}

__global__ void UpdateVoxelMapCudaKernel3(int max_index_leaf, LeafVoxelCuda* d_leaf_voxel_cuda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_index_leaf)  return;

    if (d_leaf_voxel_cuda[i].is_valid_ == 0)
    {
        int current_leaf_bottom = atomicAdd(&free_bottom_leaf_count, 1);
        free_stack_leaf[current_leaf_bottom] = i;
    }
}

__global__ void PVListUpdateCudaKernel(int feats_down_world_size, PointCloudXYZICuda* d_feats_down_body, 
                                    PointCloudXYZICuda* d_world_lidar, pointWithVarCuda* d_pv_list, Mat33* d_cross_mat_list,
                                    Mat33* d_body_cov_list)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= feats_down_world_size)  return;

    double p[3] = {d_feats_down_body[i].x, d_feats_down_body[i].y, d_feats_down_body[i].z};
    double temp[3];
    temp[0] = d_ParametersCuda.extR[0]*p[0] + d_ParametersCuda.extR[1]*p[1] + d_ParametersCuda.extR[2]*p[2] + d_ParametersCuda.extT[0];
    temp[1] = d_ParametersCuda.extR[3]*p[0] + d_ParametersCuda.extR[4]*p[1] + d_ParametersCuda.extR[5]*p[2] + d_ParametersCuda.extT[1];
    temp[2] = d_ParametersCuda.extR[6]*p[0] + d_ParametersCuda.extR[7]*p[1] + d_ParametersCuda.extR[8]*p[2] + d_ParametersCuda.extT[2];

    p[0] = d_state.rot_end[0]*temp[0] + d_state.rot_end[1]*temp[1] + d_state.rot_end[2]*temp[2] + d_state.pos_end[0];
    p[1] = d_state.rot_end[3]*temp[0] + d_state.rot_end[4]*temp[1] + d_state.rot_end[5]*temp[2] + d_state.pos_end[1];
    p[2] = d_state.rot_end[6]*temp[0] + d_state.rot_end[7]*temp[1] + d_state.rot_end[8]*temp[2] + d_state.pos_end[2];

    d_world_lidar[i].x = p[0];  d_world_lidar[i].y = p[1];  d_world_lidar[i].z = p[2];  
    d_world_lidar[i].intensity = d_feats_down_body[i].intensity;

    double cov[9], point_crossmat[9];
    for (int j=0; j<9; j++){
        cov[j] = d_body_cov_list[i].mat[j];  point_crossmat[j] = d_cross_mat_list[i].mat[j];
    }

    double rot_end_extR[9];
    rot_end_extR[0] = d_state.rot_end[0]*d_ParametersCuda.extR[0] + d_state.rot_end[1]*d_ParametersCuda.extR[3] +
                      d_state.rot_end[2]*d_ParametersCuda.extR[6];
    rot_end_extR[1] = d_state.rot_end[0]*d_ParametersCuda.extR[1] + d_state.rot_end[1]*d_ParametersCuda.extR[4] +
                      d_state.rot_end[2]*d_ParametersCuda.extR[7];
    rot_end_extR[2] = d_state.rot_end[0]*d_ParametersCuda.extR[2] + d_state.rot_end[1]*d_ParametersCuda.extR[5] +
                      d_state.rot_end[2]*d_ParametersCuda.extR[8];
    rot_end_extR[3] = d_state.rot_end[3]*d_ParametersCuda.extR[0] + d_state.rot_end[4]*d_ParametersCuda.extR[3] +
                      d_state.rot_end[5]*d_ParametersCuda.extR[6];
    rot_end_extR[4] = d_state.rot_end[3]*d_ParametersCuda.extR[1] + d_state.rot_end[4]*d_ParametersCuda.extR[4] +
                      d_state.rot_end[5]*d_ParametersCuda.extR[7];
    rot_end_extR[5] = d_state.rot_end[3]*d_ParametersCuda.extR[2] + d_state.rot_end[4]*d_ParametersCuda.extR[5] +
                      d_state.rot_end[5]*d_ParametersCuda.extR[8];
    rot_end_extR[6] = d_state.rot_end[6]*d_ParametersCuda.extR[0] + d_state.rot_end[7]*d_ParametersCuda.extR[3] +
                      d_state.rot_end[8]*d_ParametersCuda.extR[6];
    rot_end_extR[7] = d_state.rot_end[6]*d_ParametersCuda.extR[1] + d_state.rot_end[7]*d_ParametersCuda.extR[4] +
                      d_state.rot_end[8]*d_ParametersCuda.extR[7];
    rot_end_extR[8] = d_state.rot_end[6]*d_ParametersCuda.extR[2] + d_state.rot_end[7]*d_ParametersCuda.extR[5] +
                      d_state.rot_end[8]*d_ParametersCuda.extR[8];

    double temp1[9], temp2[9], temp3[9], temp4[9];
    temp1[0] = rot_end_extR[0]*cov[0] + rot_end_extR[1]*cov[3] + rot_end_extR[2]*cov[6];
    temp1[1] = rot_end_extR[0]*cov[1] + rot_end_extR[1]*cov[4] + rot_end_extR[2]*cov[7];
    temp1[2] = rot_end_extR[0]*cov[2] + rot_end_extR[1]*cov[5] + rot_end_extR[2]*cov[8];
    temp1[3] = rot_end_extR[3]*cov[0] + rot_end_extR[4]*cov[3] + rot_end_extR[5]*cov[6];
    temp1[4] = rot_end_extR[3]*cov[1] + rot_end_extR[4]*cov[4] + rot_end_extR[5]*cov[7];
    temp1[5] = rot_end_extR[3]*cov[2] + rot_end_extR[4]*cov[5] + rot_end_extR[5]*cov[8];
    temp1[6] = rot_end_extR[6]*cov[0] + rot_end_extR[7]*cov[3] + rot_end_extR[8]*cov[6];
    temp1[7] = rot_end_extR[6]*cov[1] + rot_end_extR[7]*cov[4] + rot_end_extR[8]*cov[7];
    temp1[8] = rot_end_extR[6]*cov[2] + rot_end_extR[7]*cov[5] + rot_end_extR[8]*cov[8];

    temp2[0] = temp1[0]*rot_end_extR[0] + temp1[1]*rot_end_extR[1] + temp1[2]*rot_end_extR[2];
    temp2[1] = temp1[0]*rot_end_extR[3] + temp1[1]*rot_end_extR[4] + temp1[2]*rot_end_extR[5];
    temp2[2] = temp1[0]*rot_end_extR[6] + temp1[1]*rot_end_extR[7] + temp1[2]*rot_end_extR[8];
    temp2[3] = temp1[3]*rot_end_extR[0] + temp1[4]*rot_end_extR[1] + temp1[5]*rot_end_extR[2];
    temp2[4] = temp1[3]*rot_end_extR[3] + temp1[4]*rot_end_extR[4] + temp1[5]*rot_end_extR[5];
    temp2[5] = temp1[3]*rot_end_extR[6] + temp1[4]*rot_end_extR[7] + temp1[5]*rot_end_extR[8];
    temp2[6] = temp1[6]*rot_end_extR[0] + temp1[7]*rot_end_extR[1] + temp1[8]*rot_end_extR[2];
    temp2[7] = temp1[6]*rot_end_extR[3] + temp1[7]*rot_end_extR[4] + temp1[8]*rot_end_extR[5];
    temp2[8] = temp1[6]*rot_end_extR[6] + temp1[7]*rot_end_extR[7] + temp1[8]*rot_end_extR[8];

    temp3[0] = -point_crossmat[0]*h_d_rot_var[0] - point_crossmat[1]*h_d_rot_var[3] - point_crossmat[2]*h_d_rot_var[6];
    temp3[1] = -point_crossmat[0]*h_d_rot_var[1] - point_crossmat[1]*h_d_rot_var[4] - point_crossmat[2]*h_d_rot_var[7];
    temp3[2] = -point_crossmat[0]*h_d_rot_var[2] - point_crossmat[1]*h_d_rot_var[5] - point_crossmat[2]*h_d_rot_var[8];
    temp3[3] = -point_crossmat[3]*h_d_rot_var[0] - point_crossmat[4]*h_d_rot_var[3] - point_crossmat[5]*h_d_rot_var[6];
    temp3[4] = -point_crossmat[3]*h_d_rot_var[1] - point_crossmat[4]*h_d_rot_var[4] - point_crossmat[5]*h_d_rot_var[7];
    temp3[5] = -point_crossmat[3]*h_d_rot_var[2] - point_crossmat[4]*h_d_rot_var[5] - point_crossmat[5]*h_d_rot_var[8];
    temp3[6] = -point_crossmat[6]*h_d_rot_var[0] - point_crossmat[7]*h_d_rot_var[3] - point_crossmat[8]*h_d_rot_var[6];
    temp3[7] = -point_crossmat[6]*h_d_rot_var[1] - point_crossmat[7]*h_d_rot_var[4] - point_crossmat[8]*h_d_rot_var[7];
    temp3[8] = -point_crossmat[6]*h_d_rot_var[2] - point_crossmat[7]*h_d_rot_var[5] - point_crossmat[8]*h_d_rot_var[8];

    temp4[0] = -temp3[0]*point_crossmat[0] - temp3[1]*point_crossmat[1] - temp3[2]*point_crossmat[2];
    temp4[1] = -temp3[0]*point_crossmat[3] - temp3[1]*point_crossmat[4] - temp3[2]*point_crossmat[5];
    temp4[2] = -temp3[0]*point_crossmat[6] - temp3[1]*point_crossmat[7] - temp3[2]*point_crossmat[8];
    temp4[3] = -temp3[3]*point_crossmat[0] - temp3[4]*point_crossmat[1] - temp3[5]*point_crossmat[2];
    temp4[4] = -temp3[3]*point_crossmat[3] - temp3[4]*point_crossmat[4] - temp3[5]*point_crossmat[5];
    temp4[5] = -temp3[3]*point_crossmat[6] - temp3[4]*point_crossmat[7] - temp3[5]*point_crossmat[8];
    temp4[6] = -temp3[6]*point_crossmat[0] - temp3[7]*point_crossmat[1] - temp3[8]*point_crossmat[2];
    temp4[7] = -temp3[6]*point_crossmat[3] - temp3[7]*point_crossmat[4] - temp3[8]*point_crossmat[5];
    temp4[8] = -temp3[6]*point_crossmat[6] - temp3[7]*point_crossmat[7] - temp3[8]*point_crossmat[8];

    for (int j=0; j<9; j++){
        d_pv_list[i].var[j] = temp2[j] + temp4[j] + h_d_t_var[j];
    }
}

__host__ void VoxelMapManagerCuda::Log(const double* R, double* out)
{
    double R_trace = R[3*0+0] + R[3*1+1] + R[3*2+2];
    double theta = (R_trace > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R_trace - 1));
    double Kx = R[7] - R[5], Ky = R[2] - R[6], Kz = R[3] - R[1];
    if (std::abs(theta) < 0.001)  out[0] = 0.5*Kx, out[1] = 0.5*Ky, out[2] = 0.5*Kz;
    else out[0] = 0.5*theta/std::sin(theta)*Kx, out[1] = 0.5*theta/std::sin(theta)*Ky, out[2] = 0.5*theta/std::sin(theta)*Kz;
}

__host__ void VoxelMapManagerCuda::Exp(const double v1, const double v2, const double v3, double* out)
{
    double norm = sqrt(v1*v1 + v2*v2 + v3*v3);
    double Eye3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (norm > 0.00001)
    {
        double r_ang[3] = {v1/norm, v2/norm, v3/norm};
        double K[9] = {0.0, -r_ang[2], r_ang[1], r_ang[2], 0.0, -r_ang[0], -r_ang[1], r_ang[0], 0.0};
        double KK[9];
        KK[0] = K[0]*K[0] + K[1]*K[3] + K[2]*K[6];
        KK[1] = K[0]*K[1] + K[1]*K[4] + K[2]*K[7];
        KK[2] = K[0]*K[2] + K[1]*K[5] + K[2]*K[8];
        KK[3] = K[3]*K[0] + K[4]*K[3] + K[5]*K[6];
        KK[4] = K[3]*K[1] + K[4]*K[4] + K[5]*K[7];
        KK[5] = K[3]*K[2] + K[4]*K[5] + K[5]*K[8];
        KK[6] = K[6]*K[0] + K[7]*K[3] + K[8]*K[6];
        KK[7] = K[6]*K[1] + K[7]*K[4] + K[8]*K[7];
        KK[8] = K[6]*K[2] + K[7]*K[5] + K[8]*K[8];

        for (int i=0; i<9; i++)  out[i] = Eye3[i] + std::sin(norm)*K[i] + (1.0-std::cos(norm))*KK[i];
    }
    else for (int i=0; i<9; i++)  out[i] = Eye3[i];
}

__host__ void VoxelMapManagerCuda::StateEstimationCuda(StatesGroupCuda &h_state_propagate)
{
    auto err = cudaMemcpyToSymbol(d_state_propagate, &h_state_propagate, sizeof(StatesGroupCuda));
    if (err != cudaSuccess) {
        std::cerr << "StateEstimationCuda cudaMemcpyToSymbol1 failed: "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(-1);
    }

    double dept_err = config_setting_cuda_.dept_err_;
    double beam_err = config_setting_cuda_.beam_err_;
    
    int threads1 = 256;
    int blocks1 = (feats_down_world_size_ + threads1 - 1) / threads1;
    StateEstimationCudaKernel1<<<blocks1, threads1>>>(feats_down_world_size_, d_cross_mat_list_, d_feats_down_body_,
                                                      d_body_cov_list_, dept_err, beam_err);
    err = cudaGetLastError();
    if (err != cudaSuccess)  std::cerr<<"StateEstimationCuda Launch1 error: "<<cudaGetErrorString(err)<<std::endl;
    cudaDeviceSynchronize();

    int rematch_num = 0;
    Eigen::Matrix<double, 19, 19> G, H_T_H, I_STATE;
    G.setZero();  H_T_H.setZero();  I_STATE.setIdentity();

    bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
    
    // for (int iterCount = 0; iterCount < config_setting_cuda_.max_iterations_; iterCount++)
    for (int iterCount = 0; iterCount < config_setting_cuda_.max_iterations_; iterCount++)
    {
        double total_residual = 0.0;
        effect_feat_num_ = 0;
        
        int threads2 = 256;
        int blocks2 = (feats_down_world_size_ + threads2 - 1) / threads2;
        StateEstimationCudaKernel2<<<blocks2, threads2>>>(feats_down_world_size_, d_feats_down_body_, d_world_lidar_);
        err = cudaGetLastError();
        if (err != cudaSuccess)  std::cerr<<"StateEstimationCuda Launch2 error: "<<cudaGetErrorString(err)<<std::endl;
        cudaDeviceSynchronize();

        for (int i=0; i<3; i++){
        for (int j=0; j<3; j++)  {h_d_rot_var[i*3+j] = state_.cov[i*19+j];  
                                  h_d_t_var[i*3+j] = state_.cov[(i+3)*19+j+3];}
        }

        int threads3 = 256;
        int blocks3 = (feats_down_world_size_ + threads3 - 1) / threads3;
        StateEstimationCudaKernel3<<<blocks3, threads3>>>(feats_down_world_size_, d_pv_list_, d_feats_down_body_, d_world_lidar_,
                                                          d_cross_mat_list_, d_body_cov_list_);
        err = cudaGetLastError();
        if (err != cudaSuccess)  std::cerr<<"StateEstimationCuda Launch3 error: "<<cudaGetErrorString(err)<<std::endl;
        cudaDeviceSynchronize();

        BuildResidualListOMPCuda();
        
        for (int i=0; i<effect_feat_num_; i++)  total_residual += fabs(h_ptpl_list_[i].dis_to_plane_);
        
        std::cout << "[ LIO-GPU ] downsampled feature num: " << feats_down_world_size_ << " effective feature num: " 
                  << effect_feat_num_ << " average residual: " << total_residual/effect_feat_num_ << std::endl;

        cudaMemcpy(d_ptpl_list_, h_ptpl_list_, effect_feat_num_*sizeof(PointToPlaneCuda), cudaMemcpyHostToDevice);

        int threads4 = 256;
        int blocks4 = (effect_feat_num_ + threads4 - 1) / threads4;
        StateEstimationCudaKernel4<<<blocks4, threads4>>>(effect_feat_num_, d_ptpl_list_, d_Hsub_, d_Hsub_T_R_inv_,
                                                          d_R_inv_, d_meas_vec_);
        err = cudaGetLastError();
        if (err != cudaSuccess)  std::cerr<<"StateEstimationCuda Launch4 error: "<<cudaGetErrorString(err)<<std::endl;
        cudaDeviceSynchronize();

        EKF_stop_flg = false;
        flg_EKF_converged = false;

        Eigen::Matrix<double, 6, 1> HTz;
        const double alpha = 1.0, beta = 0.0;
        cublasDgemv(cublas_handle, CUBLAS_OP_T, effect_feat_num_, 6, &alpha, d_Hsub_T_R_inv_, effect_feat_num_, d_meas_vec_, 1,
                    &beta, h_d_HTz, 1);
        cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 6, 6, effect_feat_num_, &alpha, d_Hsub_T_R_inv_, effect_feat_num_,
                    d_Hsub_, 6, &beta, h_d_Hsub_T_R_inv_Hsub, 6);
        cudaDeviceSynchronize();

        for (int i=0; i<6; i++){
            for (int j=0; j<6; j++)  H_T_H(i,j) = h_d_Hsub_T_R_inv_Hsub[i*6+j];
        }
        doubleToEigen(h_d_HTz, HTz);

        Eigen::Matrix<double, 19, 19> cov_eigen;
        doubleToEigen(state_.cov, cov_eigen);
        Eigen::Matrix<double, 19, 19> cov_eigen_transpose = cov_eigen.transpose();
        Eigen::Matrix<double, 19, 19> K_1 = (H_T_H + cov_eigen_transpose.inverse()).inverse();
        G.block<19, 6>(0, 0) = K_1.block<19, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
        Eigen::Matrix<double, 19, 1> vec;

        double rot_end_multi[9], rot_end_out[3];
        rot_end_multi[0] = state_.rot_end[0]*h_state_propagate.rot_end[0] + state_.rot_end[3]*h_state_propagate.rot_end[3] +
                           state_.rot_end[6]*h_state_propagate.rot_end[6];
        rot_end_multi[1] = state_.rot_end[0]*h_state_propagate.rot_end[1] + state_.rot_end[3]*h_state_propagate.rot_end[4] +
                           state_.rot_end[6]*h_state_propagate.rot_end[7];
        rot_end_multi[2] = state_.rot_end[0]*h_state_propagate.rot_end[2] + state_.rot_end[3]*h_state_propagate.rot_end[5] +
                           state_.rot_end[6]*h_state_propagate.rot_end[8];
        rot_end_multi[3] = state_.rot_end[1]*h_state_propagate.rot_end[0] + state_.rot_end[4]*h_state_propagate.rot_end[3] +
                           state_.rot_end[7]*h_state_propagate.rot_end[6];
        rot_end_multi[4] = state_.rot_end[1]*h_state_propagate.rot_end[1] + state_.rot_end[4]*h_state_propagate.rot_end[4] +
                           state_.rot_end[7]*h_state_propagate.rot_end[7];
        rot_end_multi[5] = state_.rot_end[1]*h_state_propagate.rot_end[2] + state_.rot_end[4]*h_state_propagate.rot_end[5] +
                           state_.rot_end[7]*h_state_propagate.rot_end[8];
        rot_end_multi[6] = state_.rot_end[2]*h_state_propagate.rot_end[0] + state_.rot_end[5]*h_state_propagate.rot_end[3] +
                           state_.rot_end[8]*h_state_propagate.rot_end[6];
        rot_end_multi[7] = state_.rot_end[2]*h_state_propagate.rot_end[1] + state_.rot_end[5]*h_state_propagate.rot_end[4] +
                           state_.rot_end[8]*h_state_propagate.rot_end[7];
        rot_end_multi[8] = state_.rot_end[2]*h_state_propagate.rot_end[2] + state_.rot_end[5]*h_state_propagate.rot_end[5] +
                           state_.rot_end[8]*h_state_propagate.rot_end[8];
        Log(rot_end_multi, rot_end_out);
        vec(0) = rot_end_out[0]; vec(1) = rot_end_out[1]; vec(2) = rot_end_out[2];
        vec(3) = h_state_propagate.pos_end[0] - state_.pos_end[0];
        vec(4) = h_state_propagate.pos_end[1] - state_.pos_end[1];
        vec(5) = h_state_propagate.pos_end[2] - state_.pos_end[2];
        vec(6) = h_state_propagate.inv_expo_time - state_.inv_expo_time;
        vec(7) = h_state_propagate.vel_end[0] - state_.vel_end[0];
        vec(8) = h_state_propagate.vel_end[1] - state_.vel_end[1];
        vec(9) = h_state_propagate.vel_end[2] - state_.vel_end[2];
        vec(10) = h_state_propagate.bias_g[0] - state_.bias_g[0];
        vec(11) = h_state_propagate.bias_g[1] - state_.bias_g[1];
        vec(12) = h_state_propagate.bias_g[2] - state_.bias_g[2];
        vec(13) = h_state_propagate.bias_a[0] - state_.bias_a[0];
        vec(14) = h_state_propagate.bias_a[1] - state_.bias_a[1];
        vec(15) = h_state_propagate.bias_a[2] - state_.bias_a[2];
        vec(16) = h_state_propagate.gravity[0] - state_.gravity[0];
        vec(17) = h_state_propagate.gravity[1] - state_.gravity[1];
        vec(18) = h_state_propagate.gravity[2] - state_.gravity[2];

        Eigen::Matrix<double, 19, 1> solution = K_1.block<19, 6>(0, 0) * HTz + vec - G.block<19, 6>(0, 0) * vec.block<6, 1>(0, 0);

        double rot_end_plus_out[9], rot_end_temp[9];
        Exp(solution(0,0), solution(1,0), solution(2,0), rot_end_plus_out);
        rot_end_temp[0] = state_.rot_end[0]*rot_end_plus_out[0] + state_.rot_end[1]*rot_end_plus_out[3] + 
                          state_.rot_end[2]*rot_end_plus_out[6];
        rot_end_temp[1] = state_.rot_end[0]*rot_end_plus_out[1] + state_.rot_end[1]*rot_end_plus_out[4] + 
                          state_.rot_end[2]*rot_end_plus_out[7];
        rot_end_temp[2] = state_.rot_end[0]*rot_end_plus_out[2] + state_.rot_end[1]*rot_end_plus_out[5] + 
                          state_.rot_end[2]*rot_end_plus_out[8];
        rot_end_temp[3] = state_.rot_end[3]*rot_end_plus_out[0] + state_.rot_end[4]*rot_end_plus_out[3] + 
                          state_.rot_end[5]*rot_end_plus_out[6];
        rot_end_temp[4] = state_.rot_end[3]*rot_end_plus_out[1] + state_.rot_end[4]*rot_end_plus_out[4] + 
                          state_.rot_end[5]*rot_end_plus_out[7];
        rot_end_temp[5] = state_.rot_end[3]*rot_end_plus_out[2] + state_.rot_end[4]*rot_end_plus_out[5] + 
                          state_.rot_end[5]*rot_end_plus_out[8];
        rot_end_temp[6] = state_.rot_end[6]*rot_end_plus_out[0] + state_.rot_end[7]*rot_end_plus_out[3] + 
                          state_.rot_end[8]*rot_end_plus_out[6];
        rot_end_temp[7] = state_.rot_end[6]*rot_end_plus_out[1] + state_.rot_end[7]*rot_end_plus_out[4] + 
                          state_.rot_end[8]*rot_end_plus_out[7];
        rot_end_temp[8] = state_.rot_end[6]*rot_end_plus_out[2] + state_.rot_end[7]*rot_end_plus_out[5] + 
                          state_.rot_end[8]*rot_end_plus_out[8];

        for (int i=0; i<9; i++)  state_.rot_end[i] = rot_end_temp[i];
        for (int i=0; i<3; i++)
        {
            state_.pos_end[i] += solution(i+3, 0);
            state_.vel_end[i] += solution(i+7, 0);
            state_.bias_g[i] += solution(i+10, 0);
            state_.bias_a[i] += solution(i+13, 0);
            state_.gravity[i] += solution(i+16, 0);
        }
        state_.inv_expo_time += solution(6, 0);

        // std::cout << "GPU state: " << std::endl;
        // std::cout << "rot end: ";
        // for (int i=0; i<9; i++)  std::cout << state_.rot_end[i] << " ";
        // std::cout << std::endl;
        // std::cout << "pos end: ";
        // for (int i=0; i<3; i++)  std::cout << state_.pos_end[i] << " ";
        // std::cout << std::endl;
        // // std::cout << "inv: " << state_.inv_expo_time << std::endl;
        // std::cout << "vel end: ";
        // for (int i=0; i<3; i++)  std::cout << state_.vel_end[i] << " ";
        // std::cout << std::endl;
        // // std::cout << "bias g: ";
        // // for (int i=0; i<3; i++)  std::cout << state_.bias_g[i] << " ";
        // // std::cout << std::endl;
        // // std::cout << "bias a: ";
        // // for (int i=0; i<3; i++)  std::cout << state_.bias_a[i] << " ";
        // // std::cout << std::endl;
        // std::cout << "gravity: ";
        // for (int i=0; i<3; i++)  std::cout << state_.gravity[i] << " ";
        // std::cout << std::endl;

        auto rot_add = solution.block<3, 1>(0, 0);
        auto t_add = solution.block<3, 1>(3, 0);
        if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015))  flg_EKF_converged = true;
        Eigen::Matrix3d rot_end_eigen;
        doubleToEigen(state_.rot_end, rot_end_eigen);
        rot_end_eigen.transposeInPlace();
        Eigen::Vector3d euler_cur = rot_end_eigen.eulerAngles(2, 1, 0);

        if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_cuda_.max_iterations_ - 2)))){
            rematch_num++;
        }

        if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_cuda_.max_iterations_ - 1)))
        {
            cov_eigen_transpose = (I_STATE - G) * cov_eigen_transpose;
            Eigen::Matrix<double, 19, 19> cov_eigen_transpose_transpose = cov_eigen_transpose.transpose();
            eigenToCuda(cov_eigen_transpose_transpose, state_.cov);
            for (int i=0; i<3; i++)  position_last_(i) = state_.pos_end[i];
            geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
            EKF_stop_flg = true;
        }

        err = cudaMemcpyToSymbol(d_state, &state_, sizeof(StatesGroupCuda));
        if (err != cudaSuccess) {
        std::cerr << "StateEstimation cudaMemcpyToSymbol2 failed: "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(-1);
        }

        if (EKF_stop_flg)  break;
    }
}

__host__ void VoxelMapManagerCuda::BuildVoxelMapCuda()
{
    float voxel_size = config_setting_cuda_.max_voxel_size_;
    float planer_threshold = config_setting_cuda_.planner_threshold_;
    double dept_err = config_setting_cuda_.dept_err_;
    double beam_err = config_setting_cuda_.beam_err_;
    auto leaf_loc = d_leaf_voxel_location_cuda;
    auto leaf_voxel = d_leaf_voxel_cuda;
    int layer_init_num[5];
    std::memcpy(layer_init_num, config_setting_cuda_.layer_init_num_.data(), 5*sizeof(int));
    
    int threads1 = 256;
    int blocks1 = (TOTAL_CAPACITY + threads1 - 1) / threads1;
    BuildVoxelMapCudaKernel1<<<blocks1, threads1>>>();

    auto err = cudaMemcpyToSymbol(d_state, &state_, sizeof(StatesGroupCuda));
    if (err != cudaSuccess) {
        std::cerr << "BuildVoxelMapCuda cudaMemcpyToSymbol1 failed: "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(-1);
    }

    ParametersCuda h_ParametersCuda;
    for (int i=0; i<9; i++)  {h_ParametersCuda.extR[i] = extR_[i];}
    for (int i=0; i<3; i++)  {h_ParametersCuda.extT[i] = extT_[i];}
    err = cudaMemcpyToSymbol(d_ParametersCuda, &h_ParametersCuda, sizeof(ParametersCuda));
    if (err != cudaSuccess){
        std::cerr << "BuildVoxelMapCuda cudaMemcpyToSymbol2 failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(-1);
    }

    auto find_ref_root = root_voxel_map_cuda.ref(cuco::find);
    auto find_ref_leaf = leaf_voxel_map_cuda.ref(cuco::find);
    auto insert_ref_root = root_voxel_map_cuda.ref(cuco::insert);
    auto insert_ref_leaf = leaf_voxel_map_cuda.ref(cuco::insert);

    int threads2 = 256;
    int blocks2 = (feats_down_world_size_ + threads1 - 1) / threads1;

    BuildVoxelMapCudaKernel2<<<blocks2, threads2>>>(feats_down_world_size_, d_feats_down_world_, d_feats_down_body_, 
                                                    d_input_points_,
                                                    d_root_voxel_location_cuda, d_root_voxel_cuda,
                                                    dept_err, beam_err, voxel_size, layer_init_num[0], layer_init_num[1],
                                                    layer_init_num[2], layer_init_num[3], layer_init_num[4],
                                                    planer_threshold,
                                                    find_ref_root, insert_ref_root);
    err = cudaGetLastError();
    if (err != cudaSuccess)  std::cerr<<"BuildVoxelMapCuda Launch2 error: "<<cudaGetErrorString(err)<<std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)  std::cerr << "BuildVoxelMapCuda Sync2 error: " << cudaGetErrorString(err) << "\n";

    root_voxel_map_cuda.for_each(
        [find_ref_leaf, insert_ref_leaf, leaf_loc, leaf_voxel]
        __device__ (cuco::pair<VOXEL_LOCATION_CUDA*, RootVoxelCuda*> slot_root)
        {
            VOXEL_LOCATION_CUDA* root_location = slot_root.first;
            RootVoxelCuda* root_value = slot_root.second;
            init_octo_tree_root_cuda(root_location, root_value, find_ref_leaf, insert_ref_leaf, 
                                leaf_loc, leaf_voxel);
        }
    );
    cudaDeviceSynchronize();

    leaf_voxel_map_cuda.for_each(
        []__device__ (cuco::pair<VOXEL_LOCATION_CUDA*, LeafVoxelCuda*> slot_leaf)
        {
            VOXEL_LOCATION_CUDA* leaf_location = slot_leaf.first;
            LeafVoxelCuda* leaf_value = slot_leaf.second;
            init_octo_tree_leaf_cuda(leaf_location, leaf_value);
        }
    );
    cudaDeviceSynchronize();
}

__host__ void VoxelMapManagerCuda::BuildResidualListOMPCuda()
{
    double voxel_size = config_setting_cuda_.max_voxel_size_;
    double sigma_num = config_setting_cuda_.sigma_num_;
    auto find_ref_root = root_voxel_map_cuda.ref(cuco::find);
    auto find_ref_leaf = leaf_voxel_map_cuda.ref(cuco::find);

    int threads1 = 256;
    int blocks1 = (feats_down_world_size_ + threads1 - 1) / threads1;
    BuildResidualListOMPCudaKernel1<<<blocks1, threads1>>>(feats_down_world_size_, d_pv_list_, d_all_ptpl_list_, d_useful_ptpl_,
                                                        find_ref_root, find_ref_leaf, voxel_size, sigma_num);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)  std::cerr<<"BuildResidualListOMPCudaKernel1 Launch error: "<<cudaGetErrorString(err)<<std::endl;
    cudaDeviceSynchronize();

    cudaMemcpy(h_all_ptpl_list_, d_all_ptpl_list_, feats_down_world_size_ * sizeof(PointToPlaneCuda), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_useful_ptpl_, d_useful_ptpl_, feats_down_world_size_ * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i=0; i<feats_down_world_size_; i++)
    {
        if (h_useful_ptpl_[i])
        {
            h_ptpl_list_[effect_feat_num_] = h_all_ptpl_list_[i];
            effect_feat_num_++;
        } 
    }
}

__host__ void VoxelMapManagerCuda::UpdateVoxelMapCuda()
{
    printf("GPU root before update: %d    leaf: %d\n", root_voxel_map_cuda.size(), leaf_voxel_map_cuda.size());
    float voxel_size = config_setting_cuda_.max_voxel_size_;
    float planer_threshold = config_setting_cuda_.planner_threshold_;
    auto leaf_loc = d_leaf_voxel_location_cuda;
    auto leaf_voxel = d_leaf_voxel_cuda;
    int layer_init_num[5];
    std::memcpy(layer_init_num, config_setting_cuda_.layer_init_num_.data(), 5*sizeof(int));

    auto find_ref_root = root_voxel_map_cuda.ref(cuco::find);
    auto find_ref_leaf = leaf_voxel_map_cuda.ref(cuco::find);
    auto insert_ref_root = root_voxel_map_cuda.ref(cuco::insert);
    auto insert_ref_leaf = leaf_voxel_map_cuda.ref(cuco::insert);

    int threads1 = 256;
    int blocks1 = (feats_down_world_size_ + threads1 - 1) / threads1;
    UpdateVoxelMapCudaKernel1<<<blocks1, threads1>>>(feats_down_world_size_, d_pv_list_, d_root_voxel_location_cuda, d_root_voxel_cuda,
                                                    voxel_size, layer_init_num[0], layer_init_num[1], layer_init_num[2], 
                                                    layer_init_num[3], layer_init_num[4], planer_threshold,
                                                    find_ref_root, insert_ref_root);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)  std::cerr<<"UpdateVoxelMapCudaKernel1 Launch error: "<<cudaGetErrorString(err)<<std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)  std::cerr << "UpdateVoxelMapCudaKernel1 Sync error: " << cudaGetErrorString(err) << "\n";

    root_voxel_map_cuda.for_each_async(
        [find_ref_leaf, insert_ref_leaf, leaf_loc, leaf_voxel]
        __device__ (cuco::pair<VOXEL_LOCATION_CUDA*, RootVoxelCuda*> slot_root)
        {
            VOXEL_LOCATION_CUDA* root_location = slot_root.first;
            RootVoxelCuda* root_value = slot_root.second;
            init_octo_tree_root_cuda(root_location, root_value, find_ref_leaf, insert_ref_leaf, 
                                leaf_loc, leaf_voxel);
        }, stream0
    );
    err = cudaStreamSynchronize(stream0);
    if (err != cudaSuccess)  std::cerr << "UpdateVoxelMapCudaKernel2 Sync error: " << cudaGetErrorString(err) << "\n";

    cudaMemcpyFromSymbolAsync(&h_free_bottom_root, free_bottom_root, sizeof(int), 0, cudaMemcpyDeviceToHost, stream0);

    if (h_free_bottom_root > max_index_root)  max_index_root = h_free_bottom_root;
    printf("GPU root after update: %d    GPU max root: %d\n", root_voxel_map_cuda.size(), max_index_root);
    h_free_bottom_root = root_voxel_map_cuda.size();

    cudaMemcpyToSymbolAsync(free_bottom_root, &h_free_bottom_root, sizeof(int), 0, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyToSymbolAsync(free_bottom_root_count, &h_free_bottom_root, sizeof(int), 0, cudaMemcpyHostToDevice, stream0);

    int threads2 = 256;
    int blocks2 = (max_index_root + threads2 - 1) / threads2;
    UpdateVoxelMapCudaKernel2<<<blocks2, threads2, 0, stream0>>>(max_index_root, d_root_voxel_cuda);
    cudaStreamSynchronize(stream0);

    cudaMemcpyAsync(h_root_voxel_location_cuda, d_root_voxel_location_cuda, max_index_root*sizeof(VOXEL_LOCATION_CUDA), 
                    cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(h_root_voxel_cuda, d_root_voxel_cuda, max_index_root*sizeof(RootVoxelCuda), 
                    cudaMemcpyDeviceToHost, stream0);
    

    leaf_voxel_map_cuda.for_each_async(
        []__device__ (cuco::pair<VOXEL_LOCATION_CUDA*, LeafVoxelCuda*> slot_leaf)
        {
            VOXEL_LOCATION_CUDA* leaf_location = slot_leaf.first;
            LeafVoxelCuda* leaf_value = slot_leaf.second;
            init_octo_tree_leaf_cuda(leaf_location, leaf_value);
        }, stream1
    );
    err = cudaStreamSynchronize(stream1);
    if (err != cudaSuccess)  std::cerr << "UpdateVoxelMapCudaKernel3 Sync error: " << cudaGetErrorString(err) << "\n";

    cudaMemcpyFromSymbolAsync(&h_free_bottom_leaf, free_bottom_leaf, sizeof(int), 0, cudaMemcpyDeviceToHost, stream1);

    if (h_free_bottom_leaf > max_index_leaf)  max_index_leaf = h_free_bottom_leaf;
    printf("GPU leaf after update: %d    GPU max leaf: %d\n", leaf_voxel_map_cuda.size(), max_index_leaf);
    h_free_bottom_leaf = leaf_voxel_map_cuda.size();

    cudaMemcpyToSymbolAsync(free_bottom_leaf, &h_free_bottom_leaf, sizeof(int), 0, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyToSymbolAsync(free_bottom_leaf_count, &h_free_bottom_leaf, sizeof(int), 0, cudaMemcpyHostToDevice, stream1);

    int threads3 = 256;
    int blocks3 = (max_index_leaf + threads2 - 1) / threads2;
    UpdateVoxelMapCudaKernel3<<<blocks2, threads2, 0, stream1>>>(max_index_leaf, d_leaf_voxel_cuda);
    cudaStreamSynchronize(stream1);

    // cudaMemcpyAsync(h_leaf_voxel_location_cuda, d_leaf_voxel_location_cuda, max_index_leaf*sizeof(VOXEL_LOCATION_CUDA), 
    //                 cudaMemcpyDeviceToHost, stream1);
    // cudaMemcpyAsync(h_leaf_voxel_cuda, d_leaf_voxel_cuda, max_index_leaf*sizeof(LeafVoxelCuda), 
    //                 cudaMemcpyDeviceToHost, stream1);
}

__host__ void VoxelMapManagerCuda::PVListUpdateCuda()
{
    int threads = 256;
    int blocks = (feats_down_world_size_ + threads - 1) / threads;
    PVListUpdateCudaKernel<<<blocks, threads>>>(feats_down_world_size_, d_feats_down_body_, d_world_lidar_, d_pv_list_, 
                                                d_cross_mat_list_, d_body_cov_list_);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)  std::cerr << "PVListUpdateCudaKernel Launch error: " << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
}

// __host__ void VoxelMapManagerCuda::mapSlidingCuda()
// {
//     if((position_last_ - last_slide_position).norm() < config_setting_cuda_.sliding_thresh)
//     {
//         std::cout << "\033[31m" << "[DEBUG]: Last sliding length cuda " << (position_last_ - last_slide_position).norm() << "\033[0m" << "\n";
//         return;
//     }

//     last_slide_position = position_last_;
//     float loc_xyz[3];
//     for (int j=0; j<3; j++)
//     {
//         loc_xyz[j] = position_last_[j] / config_setting_cuda_.max_voxel_size_;
//         if (loc_xyz[j] < 0)  loc_xyz[j] -= 1.0;
//     }

//     int x_max = (int64_t)loc_xyz[0] + config_setting_cuda_.half_map_size;
//     int x_min = (int64_t)loc_xyz[0] - config_setting_cuda_.half_map_size;
//     int y_max = (int64_t)loc_xyz[1] + config_setting_cuda_.half_map_size;
//     int y_min = (int64_t)loc_xyz[1] - config_setting_cuda_.half_map_size;
//     int z_max = (int64_t)loc_xyz[2] + config_setting_cuda_.half_map_size;
//     int z_min = (int64_t)loc_xyz[2] - config_setting_cuda_.half_map_size;

//     auto find_ref_leaf = leaf_voxel_map_cuda.ref(cuco::find);

//     root_voxel_map_cuda.for_each(
//         [find_ref_leaf, x_max, x_min, y_max, y_min, z_max, z_min]
//         __device__ (cuco::pair <VOXEL_LOCATION_CUDA*, RootVoxelCuda*> slot_root)
//         {
//             VOXEL_LOCATION_CUDA* root_location = slot_root.first;
//             RootVoxelCuda* root_value = slot_root.second;
//             clear_mem_cuda(find_ref_leaf, root_location, root_value, x_max, x_min, y_max, y_min, z_max, z_min);
//         }
//     );
//     cudaDeviceSynchronize();

//     int h_erase_count_root, h_erase_count_leaf;
//     cudaMemcpyFromSymbol(&h_erase_count_root, erase_count_root, sizeof(int), 0, cudaMemcpyDeviceToHost);
//     cudaMemcpyFromSymbol(&h_erase_count_leaf, erase_count_leaf, sizeof(int), 0, cudaMemcpyDeviceToHost);
//     root_voxel_map_cuda.erase(root_voxel_erased, root_voxel_erased+erase_count_root-1);
//     leaf_voxel_map_cuda.erase(leaf_voxel_erased, leaf_voxel_erased+erase_count_leaf-1);
//     printf("GPU root sliding: %d    leaf sliding: %d\n", h_erase_count_root-1, h_erase_count_leaf-1);
//     h_erase_count_root = 0;  h_erase_count_leaf = 0;
//     cudaMemcpyToSymbol(erase_count_root, &h_erase_count_root, sizeof(int), 0, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(erase_count_leaf, &h_erase_count_leaf, sizeof(int), 0, cudaMemcpyHostToDevice);
//     cudaDeviceSynchronize();
// }

__host__ void VoxelMapManagerCuda::VoxelMapMalloc()
{
    cudaHostAlloc(&h_feats_undistort_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda), cudaHostAllocMapped | cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_feats_down_body_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda), cudaHostAllocMapped | cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_feats_down_world_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda), cudaHostAllocMapped | cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_input_points_, TOTAL_CAPACITY * sizeof(pointWithVarCuda), cudaHostAllocMapped | cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_root_voxel_location_cuda, TOTAL_CAPACITY * sizeof(VOXEL_LOCATION_CUDA), cudaHostAllocMapped);
    cudaHostAlloc(&h_root_voxel_cuda, TOTAL_CAPACITY * sizeof(RootVoxelCuda), cudaHostAllocMapped);
    cudaHostAlloc(&h_leaf_voxel_location_cuda, TOTAL_CAPACITY * sizeof(VOXEL_LOCATION_CUDA), cudaHostAllocMapped);
    cudaHostAlloc(&h_leaf_voxel_cuda, TOTAL_CAPACITY * sizeof(LeafVoxelCuda), cudaHostAllocMapped);
    cudaHostAlloc(&h_pv_list_, TOTAL_CAPACITY * sizeof(pointWithVarCuda), cudaHostAllocMapped);
    cudaHostAlloc(&h_ptpl_list_, TOTAL_CAPACITY * sizeof(PointToPlaneCuda), cudaHostAllocMapped);
    cudaHostAlloc(&h_all_ptpl_list_, TOTAL_CAPACITY * sizeof(PointToPlaneCuda), cudaHostAllocMapped);
    cudaHostAlloc(&h_useful_ptpl_, TOTAL_CAPACITY * sizeof(bool), cudaHostAllocMapped);
    
    cudaMalloc(&d_feats_undistort_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda));
    cudaMalloc(&d_feats_down_body_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda));
    cudaMalloc(&d_feats_down_world_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda));
    cudaMalloc(&d_input_points_, TOTAL_CAPACITY * sizeof(pointWithVarCuda));
    cudaMalloc(&d_root_voxel_location_cuda, TOTAL_CAPACITY * sizeof(VOXEL_LOCATION_CUDA));
    cudaMalloc(&d_root_voxel_cuda, TOTAL_CAPACITY * sizeof(RootVoxelCuda));
    cudaMalloc(&d_leaf_voxel_location_cuda, TOTAL_CAPACITY * sizeof(VOXEL_LOCATION_CUDA));
    cudaMalloc(&d_leaf_voxel_cuda, TOTAL_CAPACITY * sizeof(LeafVoxelCuda));
    cudaMalloc(&d_cross_mat_list_, TOTAL_CAPACITY * sizeof(Mat33));
    cudaMalloc(&d_body_cov_list_, TOTAL_CAPACITY * sizeof(Mat33));
    cudaMalloc(&d_pv_list_, TOTAL_CAPACITY * sizeof(pointWithVarCuda));
    cudaMalloc(&d_ptpl_list_, TOTAL_CAPACITY * sizeof(PointToPlaneCuda));
    cudaMalloc(&d_world_lidar_, TOTAL_CAPACITY * sizeof(PointCloudXYZICuda));
    cudaMalloc(&d_all_ptpl_list_, TOTAL_CAPACITY * sizeof(PointToPlaneCuda));
    cudaMalloc(&d_useful_ptpl_, TOTAL_CAPACITY * sizeof(bool));

    cudaMalloc(&d_Hsub_, TOTAL_CAPACITY * 6 * sizeof(double));
    cudaMalloc(&d_Hsub_T_R_inv_, 6 * TOTAL_CAPACITY * sizeof(double));
    cudaMalloc(&d_R_inv_, TOTAL_CAPACITY* sizeof(double));
    cudaMalloc(&d_meas_vec_, TOTAL_CAPACITY * sizeof(double));
}

__host__ void VoxelMapManagerCuda::VoxelMapRelease()
{
    cudaFreeHost(h_feats_undistort_);
    cudaFreeHost(h_feats_down_body_);
    cudaFreeHost(h_feats_down_world_);
    cudaFreeHost(h_input_points_);
    cudaFreeHost(h_root_voxel_location_cuda);
    cudaFreeHost(h_root_voxel_cuda);
    cudaFreeHost(h_leaf_voxel_location_cuda);
    cudaFreeHost(h_leaf_voxel_cuda);
    cudaFreeHost(h_pv_list_);
    cudaFreeHost(h_ptpl_list_);
    cudaFreeHost(h_all_ptpl_list_);
    cudaFreeHost(h_useful_ptpl_);

    cudaFree(d_feats_undistort_);
    cudaFree(d_feats_down_body_);
    cudaFree(d_feats_down_world_);
    cudaFree(d_input_points_);
    cudaFree(d_root_voxel_location_cuda);
    cudaFree(d_root_voxel_cuda);
    cudaFree(d_leaf_voxel_location_cuda);
    cudaFree(d_leaf_voxel_cuda);
    cudaFree(d_cross_mat_list_);
    cudaFree(d_body_cov_list_);
    cudaFree(d_pv_list_);
    cudaFree(d_ptpl_list_);
    cudaFree(d_world_lidar_);
    cudaFree(d_all_ptpl_list_);
    cudaFree(d_useful_ptpl_);

    cudaFree(d_Hsub_);
    cudaFree(d_Hsub_T_R_inv_);
    cudaFree(d_R_inv_);
    cudaFree(d_meas_vec_);
}

__host__ void VoxelMapManagerCuda::VoxelMemCpy(int num)
{
    cudaMemcpy(d_feats_down_body_, h_feats_down_body_, num*sizeof(PointCloudXYZICuda), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feats_down_world_, h_feats_down_world_, num*sizeof(PointCloudXYZICuda), cudaMemcpyHostToDevice);
}

__host__ void VoxelMapManagerCuda::PvListCpy(int num)
{
    cudaMemcpyAsync(h_pv_list_, d_pv_list_, num*sizeof(pointWithVarCuda), cudaMemcpyDeviceToHost);
}

__host__ void VoxelMapManagerCuda::HandleCreate()
{
    cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS){
        std::cerr << "CUBLAS initialization failed!\n" << std::endl;
        std::exit(-1);
    }

    cusolver_status = cusolverDnCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS){
        std::cerr << "CUSOLVER intialization failed!\n" << std::endl;
        std::exit(-1);
    }
}

__host__ void VoxelMapManagerCuda::HandleDestroy()
{
    cublas_status = cublasDestroy(cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS){
        std::cerr << "CUBLAS destruction failed!\n" << std::endl;
        std::exit(-1);
    }

    cusolver_status = cusolverDnDestroy(cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS){
        std::cerr << "CUSOLVER destruction failed!\n" << std::endl;
        std::exit(-1);
    }
}

__host__ void VoxelMapManagerCuda::StreamCreate()
{
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
}

