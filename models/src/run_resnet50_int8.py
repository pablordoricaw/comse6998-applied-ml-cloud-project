import torch
import torchvision
from torchvision.models import resnet50
from utils import benchmark, get_imagenette_dataloader, pt_to_onnx, export_engine
import numpy as np
import os

import modelopt.torch.quantization as mtq
from onnxruntime.quantization import quantize_dynamic, QuantType

model_name = "resnet50_int8"

resnet50_model = resnet50(weights=None)
state = torch.load(f"../models/resnet50_base.pt", map_location='cuda')
resnet50_model.load_state_dict(state)
resnet50_model.to("cuda").eval()
calib_size = 256

# -- build the calibration loader from imagenette2 --
data_loader = get_imagenette_dataloader(num_samples=calib_size)

def save_calibration_npz(dataloader, output_path="calibration_data.npz", num_batches=10):
    inputs = []

    for i, (data, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        # Convert to CPU numpy
        inputs.append(data.detach().cpu().numpy())

    # Stack into a single array
    inputs = np.concatenate(inputs, axis=0)  # shape: (total_samples, 3, 224, 224)

    # Save
    np.savez(output_path, input=inputs)
    print(f"Saved calibration data to {output_path}")

save_calibration_npz(data_loader)

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
print("exporting engine...")
export_engine(model_name)
print("Running benchmark on exported engine")
benchmark_tensorrt("../models/resnet50_int8.engine", input_shape=(1, 3, 224, 224), nruns=100)
