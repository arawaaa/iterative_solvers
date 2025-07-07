#include "jacobi.h"

Jacobi::Jacobi(int process, struct neighbors adj, int dimensions[2], int global_pos[2])
    : process(process), grid(dimensions[0] + 2, std::vector<float>(dimensions[1] + 2)) // put in the boundary/shared areas
{
    if (dimensions[0] == 0 || dimensions[1] == 0)
        throw std::runtime_error("Empty dimensions prohibited.");

    boundary_conditions[left] = (neighbors[left] = adj.left) != -1;
    boundary_conditions[right] = (neighbors[right] = adj.right) != -1;
    boundary_conditions[top] = (neighbors[top] = adj.top) != -1;
    boundary_conditions[bottom] = (neighbors[bottom] = adj.bottom) != -1;
}

Jacobi::~Jacobi()
{
}

void Jacobi::exchangeData() {
    for (auto en : {left, right, top, bottom}) {
        if (boundary_conditions[en]) {
            auto [data, size] = getDataFor(en);
            MPI_Send(data, size, MPI_FLOAT, neighbors[en], 0, MPI_COMM_WORLD);
            if (en == left || en == right) free(data);
        }
    }

    for (auto en : {left, right, top, bottom}) {
        if (boundary_conditions[en]) {
            int size = 0;
            if (en == left || en == right) {
                size = grid.size();
            } else if (en == top || en == bottom) {
                size = grid[0].size();
            }
            float* recvd = static_cast<float*>(malloc(size * sizeof(float)));
            MPI_Recv(recvd, size, MPI_FLOAT, neighbors[en], 0, MPI_COMM_WORLD, nullptr);

            switch (en) {
                case left:
                    for (int i = 0; i < grid.size(); i++) {
                        grid[i][0] = recvd[i];
                    }
                case right:
                    for (int i = 0; i < grid.size(); i++) {
                        grid[i][grid[0].size() - 1] = recvd[i];
                    }
                case top:
                    grid[0].assign(recvd, recvd + size);
                    break;
                case bottom:
                    grid[grid.size() - 1].assign(recvd, recvd + size);
            }
        }
    }
}

float Jacobi::redBlackSolve()
{
    float maxErr = 0.0;
    for (int iter : {0, 1}) {
        #pragma omp parallel for reduction(max:maxErr)
        for (int i = 1; i < grid.size() - 1; i++) {
            for (int j = 1 + (i + iter) % 2; j < grid[0].size() - 1; j += 2) {
                auto prev = grid[i][j];
                grid[i][j] = (grid[i][j + 1] + grid[i][j - 1] + grid[i - 1][j] + grid[i + 1][j]) / 4.0;
                maxErr = std::max(maxErr, prev);
            }
        }
    }

    return maxErr;
}
