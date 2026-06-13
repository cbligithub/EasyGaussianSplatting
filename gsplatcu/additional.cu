
#include "kernel.cuh"
#include "matrix.cuh"

#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <torch/torch.h>
#include <tuple>

__global__ void device_cal_ddelta_dRcw(
    const int gs_num,
    const float *__restrict__ G_wi,
    const float *__restrict__ J_i,
    const float *__restrict__ Rcw,
    float *__restrict__ ddelta_dRcw)

{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    Matrix<3, 3> G_wi_mat = {
        G_wi[i * 9 + 0], G_wi[i * 9 + 1], G_wi[i * 9 + 2],
        G_wi[i * 9 + 3], G_wi[i * 9 + 4], G_wi[i * 9 + 5],
        G_wi[i * 9 + 6], G_wi[i * 9 + 7], G_wi[i * 9 + 8]};

    Matrix<2, 3> J_i_mat = {
        J_i[i * 6 + 0], J_i[i * 6 + 1], J_i[i * 6 + 2],
        J_i[i * 6 + 3], J_i[i * 6 + 4], J_i[i * 6 + 5]};

    Matrix<3, 3> Rcw_mat = {
        Rcw[i * 9 + 0], Rcw[i * 9 + 1], Rcw[i * 9 + 2],
        Rcw[i * 9 + 3], Rcw[i * 9 + 4], Rcw[i * 9 + 5],
        Rcw[i * 9 + 6], Rcw[i * 9 + 7], Rcw[i * 9 + 8]};

    Matrix<2, 3> K = J_i_mat * Rcw_mat * G_wi_mat;

    float k00 = K(0, 0);
    float k01 = K(0, 1);
    float k02 = K(0, 2);
    float k10 = K(1, 0);
    float k11 = K(1, 1);
    float k12 = K(1, 2);

    float j00 = J_i_mat(0, 0);
    float j01 = J_i_mat(0, 1);
    float j02 = J_i_mat(0, 2);
    float j10 = J_i_mat(1, 0);
    float j11 = J_i_mat(1, 1);
    float j12 = J_i_mat(1, 2);

    float g00 = G_wi_mat(0, 0);
    float g01 = G_wi_mat(0, 1);
    float g02 = G_wi_mat(0, 2);
    float g10 = G_wi_mat(1, 0);
    float g11 = G_wi_mat(1, 1);
    float g12 = G_wi_mat(1, 2);
    float g20 = G_wi_mat(2, 0);
    float g21 = G_wi_mat(2, 1);
    float g22 = G_wi_mat(2, 2);

    // row one
    ddelta_dRcw[i * 27 + 0] = 2 * (j00 * g00 * k00 + j00 * g01 * k01 + j00 * g02 * k02);
    ddelta_dRcw[i * 27 + 1] = 2 * (j00 * g10 * k00 + j00 * g11 * k01 + j00 * g12 * k02);
    ddelta_dRcw[i * 27 + 2] = 2 * (j00 * g20 * k00 + j00 * g21 * k01 + j00 * g22 * k02);
    ddelta_dRcw[i * 27 + 3] = 0;
    ddelta_dRcw[i * 27 + 4] = 0;
    ddelta_dRcw[i * 27 + 5] = 0;
    ddelta_dRcw[i * 27 + 6] = 2 * (j02 * g00 * k00 + j02 * g01 * k01 + j02 * g02 * k02);
    ddelta_dRcw[i * 27 + 7] = 2 * (j02 * g10 * k00 + j02 * g11 * k01 + j02 * g12 * k02);
    ddelta_dRcw[i * 27 + 8] = 2 * (j02 * g20 * k00 + j02 * g21 * k01 + j02 * g22 * k02);

    // row two
    ddelta_dRcw[i * 27 + 9] = j00 * g00 * k10 + j00 * g01 * k11 + j00 * g02 * k12;
    ddelta_dRcw[i * 27 + 10] = j00 * g10 * k10 + j00 * g11 * k11 + j00 * g12 * k12;
    ddelta_dRcw[i * 27 + 11] = j00 * g20 * k10 + j00 * g21 * k11 + j00 * g22 * k12;
    ddelta_dRcw[i * 27 + 12] = j11 * g00 * k00 + j11 * g01 * k01 + j11 * g02 * k02;
    ddelta_dRcw[i * 27 + 13] = j11 * g10 * k00 + j11 * g11 * k01 + j11 * g12 * k02;
    ddelta_dRcw[i * 27 + 14] = j11 * g20 * k00 + j11 * g21 * k01 + j11 * g22 * k02;
    ddelta_dRcw[i * 27 + 15] = j02 * g00 * k10 + j02 * g01 * k11 + j02 * g02 * k12 + j12 * g00 * k00 + j12 * g01 * k01 + j12 * g02 * k02;
    ddelta_dRcw[i * 27 + 16] = j02 * g10 * k10 + j02 * g11 * k11 + j02 * g12 * k12 + j12 * g10 * k00 + j12 * g11 * k01 + j12 * g12 * k02;
    ddelta_dRcw[i * 27 + 17] = j02 * g20 * k10 + j02 * g21 * k11 + j02 * g22 * k12 + j12 * g20 * k00 + j12 * g21 * k01 + j12 * g22 * k02;

    // row three
    ddelta_dRcw[i * 27 + 18] = 0;
    ddelta_dRcw[i * 27 + 19] = 0;
    ddelta_dRcw[i * 27 + 20] = 0;
    ddelta_dRcw[i * 27 + 21] = 2 * (j11 * g00 * k10 + j11 * g01 * k11 + j11 * g02 * k12);
    ddelta_dRcw[i * 27 + 22] = 2 * (j11 * g10 * k10 + j11 * g11 * k11 + j11 * g12 * k12);
    ddelta_dRcw[i * 27 + 23] = 2 * (j11 * g20 * k10 + j11 * g21 * k11 + j11 * g22 * k12);
    ddelta_dRcw[i * 27 + 24] = 2 * (j12 * g00 * k10 + j12 * g01 * k11 + j12 * g02 * k12);
    ddelta_dRcw[i * 27 + 25] = 2 * (j12 * g10 * k10 + j12 * g11 * k11 + j12 * g12 * k12);
    ddelta_dRcw[i * 27 + 26] = 2 * (j12 * g20 * k10 + j12 * g21 * k11 + j12 * g22 * k12);
}

__global__ void device_cal_dr_dq(
    const int gs_num,
    const float *__restrict__ Rcw,
    float *__restrict__ dr_dq)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    float qw, qx, qy, qz;

    Matrix<3, 3> Rcw_mat = {
        Rcw[i * 9 + 0], Rcw[i * 9 + 1], Rcw[i * 9 + 2],
        Rcw[i * 9 + 3], Rcw[i * 9 + 4], Rcw[i * 9 + 5],
        Rcw[i * 9 + 6], Rcw[i * 9 + 7], Rcw[i * 9 + 8]};

    float r00 = Rcw_mat(0, 0);
    float r01 = Rcw_mat(0, 1);
    float r02 = Rcw_mat(0, 2);
    float r10 = Rcw_mat(1, 0);
    float r11 = Rcw_mat(1, 1);
    float r12 = Rcw_mat(1, 2);
    float r20 = Rcw_mat(2, 0);
    float r21 = Rcw_mat(2, 1);
    float r22 = Rcw_mat(2, 2);

    // Convert rotation matrix to quaternion
    float trace = r00 + r11 + r22;
    if (trace > 0)
    {
        float t = sqrtf(trace + 1.0f);
        qw = 0.5 * t;
        qx = (r21 - r12) / (2.0f * t);
        qy = (r02 - r20) / (2.0f * t);
        qz = (r10 - r01) / (2.0f * t);
    }
    else
    {
        if (r00 > r11 && r00 > r22)
        {
            float t = sqrtf(1.0f + r00 - r11 - r22);
            qw = (r21 - r12) / (2.0f * t);
            qx = 0.5f * t;
            qy = (r01 + r10) / (2.0f * t);
            qz = (r02 + r20) / (2.0f * t);
        }
        else if (r11 > r22)
        {
            float t = sqrtf(1.0f + r11 - r00 - r22);
            qw = (r02 - r20) / (2.0f * t);
            qx = (r01 + r10) / (2.0f * t);
            qy = 0.5f * t;
            qz = (r12 + r21) / (2.0f * t);
        }
        else
        {
            float t = sqrtf(1.0f + r22 - r00 - r11);
            qw = (r10 - r01) / (2.0f * t);
            qx = (r02 + r20) / (2.0f * t);
            qy = (r12 + r21) / (2.0f * t);
            qz = 0.5f * t;
        }
    }

    // row one
    dr_dq[i * 36 + 0] = 2 * qw;
    dr_dq[i * 36 + 1] = 2 * qx;
    dr_dq[i * 36 + 2] = -2 * qy;
    dr_dq[i * 36 + 3] = -2 * qz;

    // row two
    dr_dq[i * 36 + 4] = -2 * qz;
    dr_dq[i * 36 + 5] = 2 * qy;
    dr_dq[i * 36 + 6] = 2 * qx;
    dr_dq[i * 36 + 7] = -2 * qw;

    // row three
    dr_dq[i * 36 + 8] = 2 * qy;
    dr_dq[i * 36 + 9] = 2 * qz;
    dr_dq[i * 36 + 10] = 2 * qw;
    dr_dq[i * 36 + 11] = 2 * qx;

    // row four
    dr_dq[i * 36 + 12] = 2 * qz;
    dr_dq[i * 36 + 13] = 2 * qw;
    dr_dq[i * 36 + 14] = 2 * qx;
    dr_dq[i * 36 + 15] = 2 * qy;

    // row five
    dr_dq[i * 36 + 16] = 2 * qw;
    dr_dq[i * 36 + 17] = -2 * qx;
    dr_dq[i * 36 + 18] = 2 * qy;
    dr_dq[i * 36 + 19] = -2 * qz;

    // row six
    dr_dq[i * 36 + 20] = -2 * qx;
    dr_dq[i * 36 + 21] = -2 * qw;
    dr_dq[i * 36 + 22] = 2 * qz;
    dr_dq[i * 36 + 23] = 2 * qy;

    // row seven
    dr_dq[i * 36 + 24] = -2 * qy;
    dr_dq[i * 36 + 25] = 2 * qz;
    dr_dq[i * 36 + 26] = -2 * qw;
    dr_dq[i * 36 + 27] = 2 * qx;

    // row eight
    dr_dq[i * 36 + 28] = 2 * qx;
    dr_dq[i * 36 + 29] = 2 * qy;
    dr_dq[i * 36 + 30] = 2 * qz;
    dr_dq[i * 36 + 31] = 2 * qw;

    // row nine
    dr_dq[i * 36 + 32] = 2 * qw;
    dr_dq[i * 36 + 33] = -2 * qx;
    dr_dq[i * 36 + 34] = -2 * qy;
    dr_dq[i * 36 + 35] = 2 * qz;
}

std::vector<torch::Tensor> cal_extra_grad(const torch::Tensor G_wi,
                                          const torch::Tensor J_i,
                                          const torch::Tensor Rcw)
{
    auto float_opts = G_wi.options().dtype(torch::kFloat32);
    auto int_opts = G_wi.options().dtype(torch::kInt32);
    int gs_num = G_wi.sizes()[0];

    torch::Tensor ddelta_dRcw = torch::full({gs_num, 3, 9}, 0.0, float_opts);

    device_cal_ddelta_dRcw<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        (float *)G_wi.contiguous().data_ptr<float>(),
        (float *)J_i.contiguous().data_ptr<float>(),
        (float *)Rcw.contiguous().data_ptr<float>(),
        (float *)ddelta_dRcw.contiguous().data_ptr<float>());
    CHECK_CUDA(DEBUG);

    torch::Tensor dr_dq = torch::full({gs_num, 9, 4}, 0.0, float_opts);
    device_cal_dr_dq<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        (float *)Rcw.contiguous().data_ptr<float>(),
        (float *)dr_dq.contiguous().data_ptr<float>());
    CHECK_CUDA(DEBUG);

    return {ddelta_dRcw, dr_dq};
}
