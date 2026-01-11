**Video Tutorial**
[Here](https://www.youtube.com/watch?v=Fxb9QSjrqsI) is the Youtube link!

**Dependencies** (use pip for Python packages, be sure to install the correct CPP packages based on your OS!):
- CPP: [LibTorch](https://pytorch.org/) C++ Library - OS dependent 
- CPP [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) - OS dependent 
- For ONNXRuntime, find the libonnxruntime shared library in: onnxruntime/lib/libonnxruntime.version.(dylib/so/dll)
- For libtorch, find the libc10 shared library in: libtorch/lib/libc10.(dylib/so/dll)
- CPP: CMake
- Python: Numpy, PyTorch, ONNX, ONNXRuntime, ONNXscript
- [Netron](https://github.com/lutzroeder/netron)

**Generate the ONNX/PT files**
- The .onnx and .pt files I included have been generated on OSX
- ONNX does support cross platform, however in my opinion it is safest to regenerate the files
- Open files: py/feedforward_nn, py/feedforward_cnn
- Uncomment the respective function: onnxExport(), or ptExport()
- Rerun those files to generate the new exported .pt, .onnx models
- Optional: Visualize the neural networks in Netron

**LIBTORCH: Build the C++ application (cpp_pt):**
- cpp_pt/CMakeLists: Change Line 5 (CMAKE_PREFIX_PATH) to where your Libtorch is extracted to
- cd cpp_pt, mkdir build, cmake .. , cmake --build .
- The Release/ folder must contain the .pt generated file(s)
- Run exe in Release/, do not forget the filename argument!

**ONNX: Build the C++ application (cpp_onnx):**
- cd cpp_onnx, mkdir build, cd build, mkdir Release
- Update the libonnxruntime file name in CMakeLists.txt line 5 (ONNX_RUNTIME_DYLIB)
- BEFORE building, the Release/ folder must contain the libonnxruntime shared library and it's symlinks (3 files)
- cd .. , cmake .. , cmake --build.
- **LINUX ONLY**: If you did not copy the symlinks, create them (only need to do this one), run commands in terminal:
    - ln -s libonnxruntime.so.1.##.# libonnxruntime.s.1
    - ln -s libonnxruntime.so.1 libonnxruntime.so
- The Release/ folder must contain the .onnx generated file(s)
- Run exe in Release/, do not forget the filename argument!

