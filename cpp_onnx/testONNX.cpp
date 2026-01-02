#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <cstring>

//cmake --build .
//cd Release
// ./test filename.onnx

//Avoid running by double clicking exe, may result in error loading .onnx file

int main(int argc, char* argv[]){

    //Check input argument
    if (argc < 2) {
        std::cerr << "Missing filename input argument! --> ./test file.onnx\n";
        return 1;
    }     

    const char* model_file = argv[1];

    //Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    //Session Options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //Create Session
    Ort::Session session(env, model_file, session_options);
    std::cout << "Model loaded successfully!" << "\n";

    //-------------------------------Ort session usage example--------------------------------//

    //Create allocator
    Ort::AllocatorWithDefaultOptions allocator;

    //Get input names
    auto inputNodeName = session.GetInputNameAllocated(0, allocator);
    const char* inputName = inputNodeName.get();
    std::cout << "Input Name: " << inputName << "\n";

    //Get input type info and shape
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();   
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Input type: " << type << "\n";
    std::vector<int64_t> input_node_dims = tensor_info.GetShape();
    std::cout << "Input shape: ";
    for (size_t i = 0; i < input_node_dims.size(); i++){
        std::cout << input_node_dims[i] << ", ";
    }
    std::cout << "\n";

    //Get output names
    auto outputNodeName = session.GetOutputNameAllocated(0, allocator);
    const char* outputName = outputNodeName.get();
    std::cout << "Output Name: " << outputName << "\n";

    //Get output type info and shape
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();   
    ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();
    std::cout << "Output type: " << output_type << "\n";    
    std::vector<int64_t> output_node_dims = output_tensor_info.GetShape();
    std::cout << "Output shape: ";
    for (size_t i = 0; i < output_node_dims.size(); i++){
        std::cout << output_node_dims[i] << ", ";
    } std::cout << "\n";

    //Input and output tensors
    std::vector<float> input_tensor;
    std::vector<int64_t> input_shape;
    std::vector<float> output_tensor;
    std::vector<int64_t> output_shape;

    if (strcmp(model_file, "testNN.onnx") == 0) {
        //For the feedforward neural net
        //input tensor (size 2)
        input_tensor = {0.4366, -0.1359};
        input_shape = {2}; 
        //output tensor (size 1)
        output_tensor = {0};
        output_shape = {1}; 

    } else {
        //For the Convolutional Neural Net
        //input tensor (size 1x3x4x4)
        input_tensor = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,              
                        8.0, 7.0, 6.0, 5.0, 8.0, 7.0, 6.0, 5.0, 8.0, 7.0, 6.0, 5.0, 8.0, 7.0, 6.0, 5.0, 
                        90.0, 10.0, 11.0, 12.0, 90.0, 10.0, 11.0, 12.0, 90.0, 10.0, 11.0, 12.0, 90.0, 10.0, 11.0, 12.0};
        input_shape = {1,3,4,4};
        //output tensor (size 6) - placeholder zeros
        output_tensor = {0,0,0,0,0,0};
        output_shape = {1,6};
    }
    
    //Create input and output tensors for inference
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_tensor.data(), 
        input_tensor.size() * sizeof(float), // Correctly calculate total size in bytes for the data buffer
        input_shape.data(), 
        input_shape.size()
    );
    
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        output_tensor.data(), 
        output_tensor.size() * sizeof(float), // Correctly calculate total size in bytes for the data buffer
        output_shape.data(), 
        output_shape.size()
    );    

    // Run inference 100 times and time it
    std::cout << "Running model..." << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i){
        try {
            session.Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, &outputTensor, 1);
        } catch (const Ort::Exception& e) {
            std::cout << "ERROR running model inference: " << e.what() << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Inference time: " << duration.count() << " milliseconds" << std::endl;    

    // Print the output value which is an array
    auto outputData = outputTensor.GetTensorMutableData<float>();
    int maxIndex = output_shape.back();
    std::cout << "Output: ";
    for (int i = 0; i < maxIndex; ++i){
        std::cout << *(outputData + i) << ", ";
    }

    std::cout << "\n Run complete" << std::endl;


    return 0;
}
