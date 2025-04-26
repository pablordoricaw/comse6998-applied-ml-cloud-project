import torch
import torchvision
import torch_tensorrt
from utils import benchmark, get_imagenette_dataloader, pt_to_onnx, export_engine

import modelopt.torch.quantization as mtq

model_name = "resnet50_int8"

resnet50_model = torch.load("../models/resnet50.pt", weights_only=False).to("cuda")
resnet50_model = resnet50_model.eval().to("cuda")
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

model_int8 = model_int8.eval().to("cuda")

# Benchmark
# TODO: not a good benchmark because needs compilation for full optimization
benchmark(model_int8, input_shape=(128, 3, 224, 224), nruns=100)

pt_to_onnx(model_name)
export_engine(model_name)
