#include "integration_alg_cpu.h"

float cpu_integrate(float *x_list, float *y_list, long long length) {
    float result = 0;

    for (int i=1; i < length; i++) {
        result += (y_list[i] + y_list[i-1]) * (x_list[i] - x_list[i-1]) / 2;
    }

    return result;

}