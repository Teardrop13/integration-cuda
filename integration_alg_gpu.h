#ifndef INTEGRATION_ALG_GPU_H
#define INTEGRATION_ALG_GPU_H

#include <iostream>
#include <fstream>

float gpu_integrate(float *x_list, float *y_list, long long length, std::ofstream& myfile);

#endif