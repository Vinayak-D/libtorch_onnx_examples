**Video WalkThrough**
- Coming soon

**Dependencies** (use pip for Python packages, be sure to install the correct CPP packages based on your OS!):
- CPP: [LibTorch](https://pytorch.org/) C++ Library - OS dependent 
- CPP [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) - OS dependent 
- For ONNXRuntime, find the libonnxruntime shared library in: onnxruntime/lib/libonnxruntime.version.(dylib/so/dll)
- For libtorch, find the libc10 shared library in: libtorch/lib/libc10.(dylib/so/dll)
- CPP: CMake
- Python: Numpy, PyTorch, ONNX, ONNXRuntime, ONNXscript
- [Netron](https://github.com/lutzroeder/netron)

**Generate the ONNX/PT files**
- The .onnx and .pt files included have been generated on: MAC OSX
- ONNX does support cross platform, however to be safe it is best to regenerate the files
- Open py/feedforward_nn, py/feedforward_cnn
- Uncomment the respective function: onnxExport(), or ptExport()
- Rerun those files to generate the new exported models
- Optional: Visualize the neural networks in Netron

**LIBTORCH: Build the C++ application:**
- cpp_pt/CMakeLists: Change Line 5 (CMAKE_PREFIX_PATH) to where your Libtorch is extracted to
- cd cpp_pt, mkdir build, cmake .. , cmake --build .
- The Release/ folder must contain the .pt generated file(s)
- run exe in Release/, do not forget the filename argument!

**ONNX: Build the C++ application:**
- cd cpp_onnx, mkdir build, cd build, mkdir Release
- BEFORE building, the Release/ folder must contain the libonnxruntime shared library 
- cd .. (one folder back), cmake .. , cmake --build.
- **LINUX ONLY**: Symlinks have to be created, (only need to do this one), run these in terminal
    - ln -s libonnxruntime.so.1.##.# libonnxruntime.s.1
    - ln -s libonnxruntime.so.1 libonnxruntime.so
- The Release/ folder must contain the .onnx generated file(s)
- Then, run exe in Release/, do not forget the filename argument!

