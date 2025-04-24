import torch
import torchvision
import torch_tensorrt
from utils import benchmark

resnet50_model = torch.load("../models/resnet50.pt", weights_only=False)

# FP32 compilation
trt_model_fp32 = torch_tensorrt.compile(
    resnet50_model,
    inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions={torch.float32},      # use a set for clarity
    workspace_size=1 << 22
)

# Benchmark
benchmark(trt_model_fp32, input_shape=(128, 3, 224, 224), nruns=100)
