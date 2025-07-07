#pragma once

#include <memory>
#include <vector>
#include <exception>
#include <cstring>

#include <omp.h>
#include <mpi.h>

class Jacobi;

struct neighbors {
    int right, left, top, bottom;
};

enum direction {
    left = 0,
    right,
    top,
    bottom
};
/**
 * Jacobi Red-Black (Gauss-Seidel)
 * We calculate roughly half of the squares from their neighbors
 * in a checkerboard pattern, and then use the updated values
 * for the remaining calculations
 */
class Jacobi
{
    int neighbors[4];
    int process;

    bool boundary_conditions[4];

    std::vector<std::vector<float>> grid;

    std::pair<void*, int> getDataFor(direction dir) {
        float* res = nullptr;
        int s = 0;
        switch (dir) {
            case left:
                res = static_cast<float*>(malloc(grid.size() * sizeof(float)));
                for (int i = 0; i < grid.size(); i++) {
                    res[i] = grid[i][1];
                }
                s = grid.size();
                break;
            case right:
                res = static_cast<float*>(malloc(grid.size() * sizeof(float)));
                for (int i = 0; i < grid.size(); i++) {
                    res[i] = grid[i][grid[0].size() - 2];
                }
                s = grid.size();
                break;
            case top:
                res = grid[1].data();
                s = grid[0].size();
                break;
            case bottom:
                res = grid[grid.size() - 2].data();
                s = grid[0].size();
        }
        return {res, s};
    }
public:
    Jacobi(int process, struct neighbors adj, int dimensions[2], int global_pos[2]);

    ~Jacobi();

    void exchangeData();

    float redBlackSolve();
};
