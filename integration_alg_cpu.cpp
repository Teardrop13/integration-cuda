#include "integration_alg_cpu.h"

float cpu_integrate(std::vector<float> x_list, std::vector<float> y_list) {
    float result = 0;

    for (int i=1; i < x_list.size(); i++) {
        result += (y_list[i] + y_list[i-1]) * (x_list[i] - x_list[i-1]) / 2;
    }

    return result;

}