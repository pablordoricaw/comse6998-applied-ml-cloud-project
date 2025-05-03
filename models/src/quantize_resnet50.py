from utils import get_imagenette_dataloader, export_engine
from modelopt.onnx.quantization.quantize import quantize
import numpy as np

calibration_loader = get_imagenette_dataloader(num_samples=1000)
calibration_samples = []

for i, (inputs, _) in enumerate(calibration_loader):
    if i >= 30:
        break
    calibration_samples.append(inputs.numpy())

calibration_data = np.concatenate(calibration_samples, axis=0)

input_path = "../models/resnet50_base.onnx"
output_path = "../models/resnet50.quant.onnx"
quantize(
    onnx_path=input_path,
    output_path=output_path,
    quantize_mode="int8",
    calibration_data=calibration_data,
    calibration_method="entropy",  # or "max"
    calibration_eps=["trt", "cuda:0", "cpu"]
)

export_engine("resnet50.quant")