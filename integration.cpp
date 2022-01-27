#include "generator.h"
#include "integration_alg_cpu.h"
#include "integration_alg_gpu.h"
#include <chrono>

#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <iostream>

#include <iostream>
#include <fstream>

using namespace std::chrono;

std::ofstream myfile;


const long long MAX_NUMBERS = 500000;
long long numbers[9] = {1000, 5000, 10000, 50000, 100000, 200000, 300000, 400000, 500000};
const float MAX = 100000;

void cpu_sort(float x_list[MAX_NUMBERS]) {
    
    for (int i=0; i<MAX_NUMBERS; i++) {
        float minimum = x_list[i];
        int minimum_index = i;
        for (int k=i; k<MAX_NUMBERS; k++) {
            if (x_list[k] < minimum) {
                minimum = x_list[k];
                minimum_index = k;
            }
        }
        float buf = x_list[i];
        x_list[i] = x_list[minimum_index];
        x_list[minimum_index] = buf;
    }
}

void print_array(float array[MAX_NUMBERS], int size) {
    for (int i=0; i<size; i++) {
        std::cout << array[i] << std::endl;
    }
}

int main() {

    myfile.open ("wyniki.csv");

    myfile << "liczba" << ',' << "czas cpu [ms]" << ',' << "czas gpu bez alokowania [ms]" << ',' << "czas gpu [ms]" << ',' << "wynik cpu" << ',' << "wynik gpu" << std::endl;

    for (int i=0; i<9; i++) {

        float* x_list = new float [numbers[i]];
        float* y_list = new float [numbers[i]];
      
        auto start = high_resolution_clock::now();
        generate(x_list, y_list, numbers[i], MAX);
        auto stop = high_resolution_clock::now();
        auto duration_generate = duration_cast<microseconds>(stop - start);
        std::cout << numbers[i] << " generated. Time: " << duration_generate.count() / 1000. << " ms" << std::endl << std::endl;

        // sorting
        std::sort(x_list, x_list+numbers[i]);
        std::cout << "list sorted" << std::endl;

        // start = high_resolution_clock::now();
        // cpu_sort(x_list);
        // stop = high_resolution_clock::now();
        // auto duration_cpu_sorting = duration_cast<microseconds>(stop - start);
        // std::cout << "cpu sorting time: " << duration_cpu_sorting.count() / 1000. << " ms" << std::endl << std::endl;

        // cpu
        start = high_resolution_clock::now();
        float result_cpu = cpu_integrate(x_list, y_list, numbers[i]);
        stop = high_resolution_clock::now();
        auto duration_cpu = duration_cast<microseconds>(stop - start);

        myfile << numbers[i] << ',' << duration_cpu.count() / 1000. << ',';
        // gpu
        start = high_resolution_clock::now();
        float result_gpu = gpu_integrate(x_list, y_list, numbers[i], myfile);
        stop = high_resolution_clock::now();
        auto duration_gpu = duration_cast<microseconds>(stop - start);

        myfile << duration_gpu.count() / 1000. << ',' << result_cpu << ',' << result_gpu << std::endl;

        // timing
        std::cout << "gpu allocating arrays + integration time: " << duration_gpu.count() / 1000. << " ms" << std::endl;
        std::cout << "cpu integration time: " << duration_cpu.count() / 1000. << " ms" << std::endl;

        // results
        std::cout << "==============================================" << std::endl;
        std::cout << "cpu result: " << result_cpu << std::endl;
        std::cout << "gpu result: " << result_gpu << std::endl;

        delete x_list;
        delete y_list;
    }

    myfile.close();
    
    return 0;
}
