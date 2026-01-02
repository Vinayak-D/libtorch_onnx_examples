cmake_minimum_required(VERSION 3.00)

project(testNN LANGUAGES CXX)
 
set(CMAKE_PREFIX_PATH "/Users/vinayakdeshpande/Documents/DATA STRUCTURES AND ALGORITHMS/C++ CODES/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/Release)

add_executable(test testPT.cpp)

target_link_libraries(test PRIVATE "${TORCH_LIBRARIES}")

set_property(TARGET test PROPERTY CXX_STANDARD 17)