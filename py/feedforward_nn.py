#Feedforward NN to ONNX test

import torch
import numpy as np
import onnx
import onnxruntime as ort
import time

example_input = torch.tensor([-0.35, 1.60], dtype = torch.float32)

#Export to ONNX
def onnxExport():  
    model.eval()    
    onnx_program = torch.onnx.export(model, example_input, dynamo=True)
    onnx_program.optimize()
    onnx_program.save("testNN.onnx")
    
#Export to .pt    
def ptExport():
    model.eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module.save("testNN.pt")

#Random seed
torch.manual_seed(0)

#neural network: n inputs, m outputs
model = torch.nn.Sequential(
    #first layer w/ and ReLU activation
    torch.nn.Linear(2,5),
    torch.nn.ReLU(),
    #second layer same as first
    torch.nn.Linear(5,5), 
    torch.nn.ReLU(),
    #last layer, linear
    torch.nn.Linear(5,1)
    )

#Inference
temp = np.zeros((1,2))
temp[0][0] = 0.4366
temp[0][1] = -0.1359
test_input = torch.tensor(temp, dtype=torch.float32)

output = model(test_input)
print ("Inference: ", model(test_input))


#Export
#ptExport()
#onnxExport()

#Load ONNX
# onnx_model = onnx.load("testNN.onnx")
# onnx.checker.check_model(onnx_model)


x = np.random.rand(2).astype(np.float32)
ort_sess = ort.InferenceSession('testNN.onnx')

#Time 100 inference runs
start_time = time.perf_counter()
for i in range(1000):
    outputs = ort_sess.run(None, {'input': x})
end_time = time.perf_counter()

duration_ms = (end_time - start_time) * 1000

print("ONNX runtime: ", outputs, "time: ", duration_ms, " ms")
