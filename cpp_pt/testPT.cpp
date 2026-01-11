#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>


//cmake --build .
//cd Release
// ./test filename.pt

//Avoid running by double clicking exe, may result in error loading .pt file

int main(int argc, char* argv[]){

    //Check input argument
    if (argc < 2) {
        std::cerr << "Missing filename input argument! --> ./test file.pt\n";
        return 1;
    }    

    //Load .pt file
    std::string filename = argv[1];
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(filename); // Load the saved .pt file
        std::cout << "Model loaded successfully.\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    //Turn off training mode
    module.eval();
    std::cout << "Training mode turned off.\n";

    // Prepare input tensor for NN or CNN
    torch::Tensor input_tensor;
    
    if (filename == "testNN.pt"){
        //For the feedforward neural net
        input_tensor = torch::randn({1, 2}); 

    } else{
        //For the convolutional neural net
        input_tensor = torch::zeros({1, 3, 4, 4}, torch::kFloat32);
        // R channel (channel 0)
        input_tensor[0][0] = torch::tensor({
            {1.f, 2.f, 3.f, 4.f},
            {1.f, 2.f, 3.f, 4.f},
            {1.f, 2.f, 3.f, 4.f},
            {1.f, 2.f, 3.f, 4.f}
        });

        // G channel (channel 1)
        input_tensor[0][1] = torch::tensor({
            {8.f, 7.f, 6.f, 5.f},
            {8.f, 7.f, 6.f, 5.f},
            {8.f, 7.f, 6.f, 5.f},
            {8.f, 7.f, 6.f, 5.f}
        });

        // B channel (channel 2)
        input_tensor[0][2] = torch::tensor({
            {90.f, 10.f, 11.f, 12.f},
            {90.f, 10.f, 11.f, 12.f},
            {90.f, 10.f, 11.f, 12.f},
            {90.f, 10.f, 11.f, 12.f}
        });
    }

    // Create a random input tensor matching the expected input shape
    std::cout << "Input tensor: " << input_tensor << "\n";  
    std::cout << "Input tensor size: " << input_tensor.sizes() << "\n";

    // Create a vector of IValue (input for the model)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Run inference 1000 times and time it
    torch::Tensor output;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i){
        output = module.forward(inputs).toTensor();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Inference time: " << duration.count() << " milliseconds" << std::endl;
    
    // Print output which is a tensor
    std::cout << "Output tensor: " << output << "\t of size: " << output.sizes() << std::endl;
    std::cout << "Inference performed successfully!\n" << std::endl;   

    return 0;
}
