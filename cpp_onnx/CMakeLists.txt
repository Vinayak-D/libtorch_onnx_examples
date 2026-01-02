cmake_minimum_required(VERSION 3.00)

project(testNN LANGUAGES CXX)

set(ONNX_RUNTIME_DYLIB "${PROJECT_SOURCE_DIR}/build/Release/libonnxruntime.1.23.2.dylib")
add_library(onnxruntime_1_23_2 SHARED IMPORTED)
set_target_properties(onnxruntime_1_23_2 PROPERTIES IMPORTED_LOCATION ${ONNX_RUNTIME_DYLIB})
target_include_directories(onnxruntime_1_23_2 INTERFACE ${PROJECT_SOURCE_DIR}/include)    

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/Release)

add_executable(test testONNX.cpp)

target_link_libraries(test PRIVATE onnxruntime_1_23_2)

set_property(TARGET test PROPERTY CXX_STANDARD 17)