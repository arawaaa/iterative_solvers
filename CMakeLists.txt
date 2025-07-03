cmake_minimum_required(VERSION 3.0)

project(iterative_solvers)

add_executable(iterative_solvers main.cpp)

install(TARGETS iterative_solvers RUNTIME DESTINATION bin)
