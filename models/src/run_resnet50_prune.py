import torch
import torchvision
from torchvision.models import resnet50
from utils import benchmark, pt_to_onnx, export_engine, get_imagenette_dataloader
import modelopt.torch.prune.pruning as mtp

model_name = "resnet50_pruned"

# -- Load model --
resnet50_model = resnet50(weights=None)
state = torch.load("../models/resnet50_base.pt", map_location="cuda")
resnet50_model.load_state_dict(state)
resnet50_model.to("cuda").eval()


train_loader = get_imagenette_dataloader(num_samples=1500)
validation_loader = get_imagenette_dataloader(num_samples=500)

# -- Create dummy input -- 
dummy_input = torch.randn(1, 3, 224, 224).to("cuda")

# -- Define pruning settings --
def score_func(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

mode = "fastnas"
constraints = {"flops": "70%"}
config = {
    "verbose": True,
    "score_func": score_func,
    "data_loader": train_loader,
}

# -- Prune model --
pruned_model, pruning_state = mtp.prune(
    model=resnet50_model,
    mode=mode,
    constraints=constraints,
    dummy_input=dummy_input,
    config=config
)

pruned_model.to("cuda").eval()

# -- Benchmark pruned model --
benchmark(pruned_model, input_shape=(1, 3, 224, 224), nruns=100)

# -- Export pruned model --
pt_to_onnx(pruned_model, model_name)
print("exporting engine...")
export_engine(model_name)


