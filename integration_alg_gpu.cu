#include "integration_alg_gpu.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


long maxGridSize;
long maxThreadsPerBlock;

float *d_x_list;
float *d_y_list;
int *d_length;
float *d_result;
float *d_result_list;

__global__ void integrate(float *x_array, float *y_array, int *length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < *(length-1)) {
        d_result_list[i] = (y_list[i] + y_list[i+1]) * (x_list[i+1] - x_list[i]) / 2
    }
}

__global__ void sum_array(float *array, int *length, float *result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < *length) {
        atomicAdd(result, array[i]);
    }
}

void cuda_initialize() {
    int device = 0;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    maxGridSize = deviceProp.maxGridSize[0];
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    std::cout << "==============================================" << std::endl;
    std::cout << "Max dimension size of a grid size (x): " << maxGridSize << std::endl;
    std::cout << "Maximum number of threads per block: " << maxThreadsPerBlock << std::endl;
    std::cout << "==============================================" << std::endl << std::endl;
}

void cuda_clean() {
    cudaFree(d_x_list);
    cudaFree(d_y_list);
    cudaFree(d_length);
    cudaFree(d_result);
}

float gpu_integrate(float *x_list, float *y_list, int length) {
    cuda_initialize();

    float result = 0;

    cudaMalloc((void **)&d_x_list, sizeof(float) * length);
    cudaMemcpy(d_x_list, &x_list, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_y_list, sizeof(float) * length);
    cudaMemcpy(d_y_list, &y_list, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result_list, sizeof(float) * (length - 1));

    cudaMalloc((void **)&d_length, sizeof(int));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    integrate<<>>>(d_x_list, d_y_list, d_length);

    cudaDeviceSynchronize();

    sum_array<<>>>(d_result_list, d_length, d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(result, &d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cuda_clean();

    return result;
}