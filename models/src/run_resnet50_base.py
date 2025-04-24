import torch
from utils import benchmark

resnet50_model = torch.load("../models/resnet50.pt", weights_only=False)

# Model benchmark without Torch-TensorRT
model = resnet50_model.eval().to("cuda")
benchmark(model, input_shape=(128, 3, 224, 224), nruns=100)

