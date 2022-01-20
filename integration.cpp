#include <iostream>
#include <algorithm>

#include "generator.h"
#include "integration_alg_cpu.h"
#include "integration_alg_gpu.h"

const int NUMBERS = 1000;
const float MAX = 10000000;

int main()
{

    std::vector<float> x_list;
    std::vector<float> y_list;

    generate(&x_list, &y_list, NUMBERS, MAX);

    std::sort(x_list.begin(), x_list.end());

    for (int i=0; i< NUMBERS; i++) {
        std::cout << x_list[i] << std::endl;
    }

    float result = cpu_integrate(x_list, y_list);

    std::cout << result << std::endl;

    return 0;
}
