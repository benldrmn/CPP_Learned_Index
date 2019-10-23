cmake_minimum_required(VERSION 3.10)
project(learned_index)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS -pthread)

add_subdirectory(tiny_dnn)
add_subdirectory(btree)
add_executable(learned_index main.cpp learned_index.h models/nn_model.h models/btree_model.h records.h)
target_link_libraries(learned_index tiny_dnn cppbtree)
