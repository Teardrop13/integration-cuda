#include <iostream>
#include <algorithm>

#include "generator.h"
#include "integration_alg_cpu.h"
#include "integration_alg_gpu.h"

const int NUMBERS = 1000;
const float MAX = 100000;

int main()
{

    float x_list[NUMBERS];
    float y_list[NUMBERS];

    generate(x_list, y_list, NUMBERS, MAX);

    std::sort(std::begin(x_list), std::end(x_list));

    for (int i=0; i< NUMBERS; i++) {
        std::cout << x_list[i] << std::endl;
    }

    // float x_list[] = {0,1,2};
    // float y_list[] = {1,2,2};

    float result = cpu_integrate(x_list, y_list, NUMBERS);

    std::cout << result << std::endl;

    return 0;
}
