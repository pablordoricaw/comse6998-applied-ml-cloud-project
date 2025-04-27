import torch
import torchvision
import torch_tensorrt
from torchvision.models import resnet50
from utils import benchmark, get_imagenette_dataloader, pt_to_onnx, export_engine

import modelopt.torch.quantization as mtq

model_name = "resnet50_int8"

resnet50_model = resnet50(weights=None)
state = torch.load(f"../models/resnet50_base.pt", map_location='cuda')
resnet50_model.load_state_dict(state)
resnet50_model.to("cuda").eval()
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

model_int8.to("cuda").eval()
# torch.save(model_int8, '../models/resnet50_base.pt')

# Benchmark
# TODO: not a good benchmark because needs compilation for full optimization
benchmark(model_int8, input_shape=(1, 3, 224, 224), nruns=100)

pt_to_onnx(model_int8, model_name)
export_engine(model_name)
