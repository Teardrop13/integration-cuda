#include "generator.h"
#include "integration_alg_cpu.h"
#include "integration_alg_gpu.h"
#include <chrono>

#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <iostream>


using namespace std::chrono;

const long long NUMBERS = 500000;
const float MAX = 100000;

int main()
{

    float x_list[NUMBERS];
    float y_list[NUMBERS];

    auto start = high_resolution_clock::now();
    generate(x_list, y_list, NUMBERS, MAX);
    auto stop = high_resolution_clock::now();
    auto duration_generate = duration_cast<microseconds>(stop - start);
    std::cout << NUMBERS << " generated. Time: " << duration_generate.count() / 1000. << " ms" << std::endl;

    // sorting
    std::sort(std::begin(x_list), std::end(x_list));
    std::cout << "list sorted" << std::endl;

    // cpu
    start = high_resolution_clock::now();
    float result_cpu = cpu_integrate(x_list, y_list, NUMBERS);
    stop = high_resolution_clock::now();
    auto duration_cpu = duration_cast<microseconds>(stop - start);

    // gpu
    start = high_resolution_clock::now();
    float result_gpu = gpu_integrate(x_list, y_list, NUMBERS);
    stop = high_resolution_clock::now();
    auto duration_gpu = duration_cast<microseconds>(stop - start);

    // timing
    std::cout << "gpu allocating arrays + integration time: " << duration_gpu.count() / 1000. << " ms" << std::endl;
    std::cout << "cpu integration time: " << duration_cpu.count() / 1000. << " ms" << std::endl;

    // results
    std::cout << "==============================================" << std::endl;
    std::cout << "cpu result: " << result_cpu << std::endl;
    std::cout << "gpu result: " << result_gpu << std::endl;
    return 0;
}
