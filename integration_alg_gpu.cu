#include "integration_alg_gpu.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


long maxGridSize;
long maxThreadsPerBlock;

float *d_x_array;
float *d_y_array;
long long *d_length;
float *d_result;

__global__ void integrate(float *x_array, float *y_array, long long *length, float *result) {

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
    cudaFree(d_x_array);
    cudaFree(d_y_array);
    cudaFree(d_length);
    cudaFree(d_result);
}

float gpu_integrate(std::vector<float> x_list, std::vector<float> y_list) {
    cuda_initialize();

    float *x_array = &x_list[0];

    long long length = x_list.size()
    float result = 0;

    cudaMalloc((void **)&d_x_array, sizeof(float) * length);
    cudaMemcpy(d_x_array, &x_array, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_y_array, sizeof(float) * length);
    cudaMemcpy(d_y_array, &y_array, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_length, sizeof(long long));
    cudaMemcpy(d_length, &length, sizeof(long long), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    integrate<<>>>(d_x_list, d_y_list, d_length, d_result);

    cuda_clean();

    return result;
}