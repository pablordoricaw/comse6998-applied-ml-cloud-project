import torch
from torchvision.models import ResNet50_Weights

resnet50_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
resnet50_model.eval()

torch.save(resnet50_model.state_dict(), '../models/resnet50_base.pt')

