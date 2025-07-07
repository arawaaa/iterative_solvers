cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 23)

project(iterative_solvers)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(iterative_solvers main.cpp jacobi.cpp)
target_link_libraries(iterative_solvers PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)

install(TARGETS iterative_solvers RUNTIME DESTINATION bin)
