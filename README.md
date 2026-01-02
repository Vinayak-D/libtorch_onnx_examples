**Video WalkThrough**
- Coming soon

**Dependencies** (use pip for Python packages, be sure to install the correct CPP packages based on your OS!):
- CPP: [LibTorch](https://pytorch.org/) C++ Library - OS dependent 
- CPP [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) - OS dependent 
- For ONNXRuntime, find the libonnxruntime shared library in: onnxruntime/lib/libonnxruntime.version.(dylib/so/dll)
- For libtorch, find the libc10 shared library in: libtorch/lib/libc10.(dylib/so/dll)
- CPP: CMake
- Python: Numpy, PyTorch, ONNX, ONNXRuntime
- [Netron](https://github.com/lutzroeder/netron)

**Generate the ONNX/PT files**
- The .onnx and .pt files included have been generated on: MAC OSX
- Open py/feedforward_nn, py/feedforward_cnn
- Uncomment the respective function: onnxExport(), or ptExport()
- Rerun those files to generate the new exported models
- Optional: Visualize the neural networks in Netron

**Build the C++ application:**
Run these commands (tested on Win/Mac/Linux):
- cd cpp_onnx / cpp_pt
- mkdir build
- cmake ..
- cmake --build .
- For ONNX, the build/Release folder must contain the .onnx generated file, and libonnxruntime.#.#.# shared library 
- For PT, the build/Release folder must contain the .pt generated file
- run exe in build/Release, ideally from command line

**NOTE:** If you get a libTorch related error, be sure you have downloaded **and extracted** the libtorch-#.#.# zip folder, it does not matter where the folder is located.
