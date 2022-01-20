#include <math.h>
#include <stdio.h>
#include <algorithm>


void generate(float *x_list, float *y_list, const int number, const float max) {
    for (int i = 0; i < number; i++) {
        float x = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max));

        if (std::find(x_list, x_list+number,x)!=x_list+number) {
            i--;
        } else {
            x_list[i] = x;
            y_list[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max));
        }
    }
}