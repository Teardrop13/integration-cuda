#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>

const float MAX = 100000;

void generate(std::vector<float> *x_list, std::vector<float> *y_list, const int number) {
    for (int i = 0; i < number; i++) {
        float x = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX));

        if (std::find((*x_list).begin(), (*x_list).end(),x)!=(*x_list).end()) {
            i--;
        } else {
            (*x_list).push_back(x);
            (*y_list).push_back(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX)));
        }
    }

    std::sort((*x_list).begin(), (*x_list).end());
}