#include "../generator/generator.h"
#include <iostream>
#include <iomanip> 

int main()
{
    const int numbers = 1000;

    std::vector<float> x_list;
    std::vector<float> y_list;

    generate(&x_list, &y_list, numbers);

    for (int i=0; i< numbers; i++) {
        std::cout << x_list[i] << std::endl;
    }

    return 0;
}
