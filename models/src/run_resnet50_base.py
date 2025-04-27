import torch
from torchvision.models import resnet50
from utils import benchmark, pt_to_onnx, export_engine

model_name = "resnet50_base"
resnet50_model = resnet50(weights=None)
state = torch.load(f"../models/{model_name}.pt", map_location='cuda')
resnet50_model.load_state_dict(state)
resnet50_model.to("cuda").eval()

# Model benchmark without Torch-TensorRT
benchmark(resnet50_model, input_shape=(1, 3, 224, 224), nruns=100)

pt_to_onnx(resnet50_model, model_name)
export_engine(model_name)
