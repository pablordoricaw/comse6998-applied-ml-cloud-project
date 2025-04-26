import torch
from utils import benchmark, pt_to_onnx, export_engine

model_name = "resnet50_base"
resnet50_model = torch.load(f"../models/{model_name}.pt", weights_only=False).to("cuda")

# Model benchmark without Torch-TensorRT
model = resnet50_model.eval().to("cuda")
benchmark(model, input_shape=(1, 3, 224, 224), nruns=100)

pt_to_onnx(model_name)
export_engine(model_name)
