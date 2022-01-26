#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

long maxGridSize;
long maxThreadsPerBlock;

float *d_x_list;
float *d_y_list;
long long *d_length;
float *d_result;
float *d_result_list;

__global__ void integrate(float *d_x_list, float *d_y_list, float *d_result_list, long long *d_length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < *(d_length)-1) {
        d_result_list[i] = (d_y_list[i] + d_y_list[i+1]) * (d_x_list[i+1] - d_x_list[i]) / 2;
    }
}

__global__ void sum_array(float *d_list, long long *d_length, float *d_result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < (*d_length)-1) {
        atomicAdd(d_result, d_list[i]);
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
    cudaFree(d_result_list);
    cudaFree(d_length);
    cudaFree(d_result);
}

float gpu_integrate(float *x_list, float *y_list, long long length) {
    cuda_initialize();

    float result = 0;

    cudaMalloc((void **)&d_x_list, sizeof(long long) * length);
    cudaMemcpy(d_x_list, x_list, sizeof(long long) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_y_list, sizeof(float) * length);
    cudaMemcpy(d_y_list, y_list, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result_list, sizeof(float) * (length - 1));

    cudaMalloc((void **)&d_length, sizeof(long long));
    cudaMemcpy(d_length, &length, sizeof(long long), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    long long threads = 256;
    long long blocks = ceil(1.* length / threads);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    integrate<<<blocks, threads>>>(d_x_list, d_y_list, d_result_list, d_length);

    cudaDeviceSynchronize();

    sum_array<<<blocks, threads>>>(d_result_list, d_length, d_result);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "gpu integration time: " << time << "ms" << std::endl;

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cuda_clean();

    return result;
}