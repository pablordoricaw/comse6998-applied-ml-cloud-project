import torch
import torchvision
import torch_tensorrt
from utils import benchmark

import modelopt.torch.quantization as mtq

resnet50_model = torch.load("../models/resnet50.pt", weights_only=False)
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
    model,
    mtq.INT8_SMOOTHQUANT_CFG,
    forward_loop
)

# now `model_int8` is your INT8‐calibrated ResNet-50
torch.save(model_int8.state_dict(), "resnet50_int8.pt")

# FP32 compilation
#trt_model_fp32 = torch_tensorrt.compile(
#    resnet50_model,
#    inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
#    enabled_precisions={torch.float32},      # use a set for clarity
#    workspace_size=1 << 22
#)

# Benchmark
benchmark(model_int8, input_shape=(128, 3, 224, 224), nruns=100)
