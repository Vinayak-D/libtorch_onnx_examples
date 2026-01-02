#Feedforward CNN to ONNX test

import torch
import numpy as np
import onnx
import onnxruntime as ort
import time

image_np = np.array([[[[1.0,2.0,3.0,4.0]]*4, #R
                     [[8.0,7.0,6.0,5.0]]*4, #G
                     [[90.0,10.0,11.0,12.0]]*4, #B
                     ]], dtype=np.float32)
#batch size: 1 
#already: 1x3x4x4, no need to permute
example_input = torch.tensor(image_np, dtype=torch.float32)

print("Input tensor: ")
print("R Channel 1:", image_np[0,0,:,:])
print("G Channel 1:", image_np[0,1,:,:])
print("B Channel 1:", image_np[0,2,:,:])

#Export to ONNX
def onnxExport():   
    model.eval()   
    onnx_program = torch.onnx.export(model, example_input, dynamo=True)
    onnx_program.optimize()
    onnx_program.save("testCNN.onnx")
    
#Export to .pt    
def ptExport():
    model.eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module.save("testCNN.pt")

#Random seed
torch.manual_seed(0)

#CNN
model = torch.nn.Sequential(    
    #Pad input image (P = 2)
    torch.nn.ZeroPad2d(2),
    # Conv2D with 16 2x2 filters and stride of 1
    # Input image has 3 RGB Channels, 16 filters and size 8
    torch.nn.Conv2d(3, 16, 2),
    # Batch Normalization
    torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
    # ReLU
    torch.nn.ReLU(),
    # Max Pooling 2D with filter size = 2 and stride = 2
    torch.nn.MaxPool2d(2,2,0),
    # Flatten layer
    torch.nn.Flatten(),
    # # Last layer (6)
    torch.nn.Linear(144, 6),
    # # Softmax activation (multiclass)
    torch.nn.Sigmoid()   
    )

#Inference
print("Inference test: ", model(example_input))

#---Export
#ptExport()
#onnxExport()

#ONNX load and run test
x = image_np

ort_sess = ort.InferenceSession('testCNN.onnx')

#Time 100 inference runs
start_time = time.perf_counter()

for i in range(1000):
    outputs = ort_sess.run(None, {'input': x})

end_time = time.perf_counter()

duration_ms = (end_time - start_time) * 1000

print("ONNX runtime: ", outputs, "time: ", duration_ms, " ms")