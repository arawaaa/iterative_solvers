#include <iostream>
#include <mpi.h>
#include <cmath>

#include "jacobi.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    std::pair<int, int> grid_size(100, 100); // row, col
    int num_process = 0;
    int own_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    auto tiles_num = std::sqrt(num_process);
    if (tiles_num*tiles_num != num_process)
        return 1;

    int tiles_int = static_cast<int>(tiles_num);
    int per_tile_row = get<0>(grid_size) / tiles_int;
    int row_extra = get<0>(grid_size) % tiles_int;
    int per_tile_col = get<1>(grid_size) / tiles_int;
    int col_extra = get<1>(grid_size) % tiles_int;

    int row_idx = own_rank / tiles_int;
    int col_idx = own_rank % tiles_int;

    // The grids are flipped, i.e. going down row is moving up in the plane
    neighbors adj {
        col_idx == tiles_int - 1 ? -1 : own_rank + 1,
        col_idx == 0 ? -1 : own_rank - 1,
        row_idx == 0 ? -1 : own_rank - tiles_int,
        row_idx == tiles_int - 1 ? -1 : own_rank + tiles_int
    };

    Jacobi fs(0, adj, {
        per_tile_row + (col_idx < row_extra ? 1 : 0),
        per_tile_col + (row_idx < col_extra ? 1 : 0)
    }, {0, 0});

    MPI_Finalize();
    return 0;
}
