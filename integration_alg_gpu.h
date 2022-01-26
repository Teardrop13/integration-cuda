#ifndef INTEGRATION_ALG_GPU_H
#define INTEGRATION_ALG_GPU_H

#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

float gpu_integrate(float *x_list, float *y_list, long long length);

#endif