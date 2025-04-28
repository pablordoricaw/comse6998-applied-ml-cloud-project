import subprocess
import sys
import torch
from pathlib import Path
from torchvision.models import ResNet50_Weights

target_dir = Path('../data/imagenette_calibration')
target_dir.mkdir(parents=True, exist_ok=True)

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
archive_name = "imagenette2.tgz"

try:
    # Download the archive
    subprocess.run(
        ["wget", url, "-O", archive_name],
        cwd=target_dir,
        check=True
    )
    print(f"Downloaded {archive_name} into {target_dir}")

    # Extract the archive
    subprocess.run(
        ["tar", "-xvf", archive_name],
        cwd=target_dir,
        check=True
    )
    print(f"Extracted {archive_name} in {target_dir}")
except subprocess.CalledProcessError as e:
    print(f"Command '{e.cmd}' failed with exit code {e.returncode}", file=sys.stderr)
    sys.exit(e.returncode)

# load resnet50
print("load resnet50 model")
resnet50_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
resnet50_model.eval()

#save resnet50
print("saving resnet50 model")
Path("../models").mkdir(parents=True, exist_ok=True)
torch.save(resnet50_model.state_dict(), '../models/resnet50_base.pt')
print("done.")
