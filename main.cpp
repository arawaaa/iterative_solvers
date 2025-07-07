#include <iostream>

#include "jacobi.h"

int main(int argc, char **argv) {
    int grid_size[2] = {100, 100};
    int origin[2] = {0, 0};

    Jacobi fs(0, {-1, -1, -1, -1}, grid_size, origin);

    fs.redBlackSolve();
    return 0;
}
