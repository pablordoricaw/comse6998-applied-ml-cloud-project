import torch
import torchvision
from utils import benchmark, get_imagenette_dataloader

import modelopt.torch.quantization as mtq

device = torch.device("cuda")
resnet50_model = torch.load("../models/resnet50.pt", weights_only=False).to(device)
calib_size = 256

# -- build the calibration loader from imagenette2 --
data_loader = get_imagenette_dataloader(num_samples=calib_size)

# -- define the forward pass that feeds your calibration data --
def forward_loop(model):
    model.eval()
    with torch.no_grad():
        for xb, _ in data_loader:             # xb = images, ignore labels
            xb = xb.to(next(model.parameters()).device)
            _ = model(xb)

# -- run PTQ calibration --
model_int8 = mtq.quantize(
    resnet50_model,
    mtq.INT8_SMOOTHQUANT_CFG,
    forward_loop
)

mode_int8 = model_int8.to(device)
model_int8.eval()

# now `model_int8` is your INT8‐calibrated ResNet-50
inputs = torch.randn(128, 3, 224, 224, device=device)
torch.onnx.export(model_int8, inputs, "../models/resnet50_int8.onnx")

# Benchmark
benchmark(model_int8, input_shape=(128, 3, 224, 224), nruns=100)
