import torch
import torchvision
from torchvision.models import resnet50
from utils import benchmark, pt_to_onnx, export_engine, get_imagenette_dataloader
import modelopt.torch.sparsity.sparsification as mts

model_name = "resnet50_sparse"

# Load model
model = resnet50(weights=None)
state = torch.load("../models/resnet50_base.pt", map_location="cuda")
model.load_state_dict(state)
model.to("cuda").eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224).to("cuda")

# sparsify
sparse_model = mts.sparsify(
    model=model,
    mode="sparsegpt",
    config={"data_loader": get_imagenette_dataloader(num_samples=1000)},
)
sparse_model.to("cuda").eval()

# Benchmark
benchmark(sparse_model, input_shape=(1, 3, 224, 224), nruns=100)

# Export
pt_to_onnx(sparse_model, model_name)
print("exporting engine...")
export_engine(model_name)

