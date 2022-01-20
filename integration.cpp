#include <iostream>
#include <algorithm>
#include <chrono>
#include "generator.h"
#include "integration_alg_cpu.h"
#include "integration_alg_gpu.h"

using namespace std::chrono;

const long long NUMBERS = 50000;
const float MAX = 100000;

int main()
{

    float x_list[NUMBERS];
    float y_list[NUMBERS];

    generate(x_list, y_list, NUMBERS, MAX);

    std::cout << "numbers generated" << std::endl;

    // sorting
    std::sort(std::begin(x_list), std::end(x_list));
    std::cout << "list sorted" << std::endl;

    // cpu
    auto start = high_resolution_clock::now();
    float result = cpu_integrate(x_list, y_list, NUMBERS);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // gpu
    auto start2 = high_resolution_clock::now();
    float result2 = gpu_integrate(x_list, y_list, NUMBERS);
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);

    // results
    std::cout << result << " time: " << duration.count() / 1000. << " ms" << std::endl;
    std::cout << result2 << " time: " << duration2.count() / 1000. << " ms" << std::endl;

    return 0;
}
