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

model_int8.eval()

# now `model_int8` is your INT8‐calibrated ResNet-50
torch.save(model_int8.state_dict(), "../models/resnet50_int8.pt")

# Benchmark
benchmark(model_int8, input_shape=(128, 3, 224, 224), nruns=100)
