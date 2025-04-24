import torch
from PIL import Image   # don't know why this import isn't working
from utils import predict

resnet50_model = torch.load("../models/resnet50.pt", weights_only=False)
resnet50_model.eval()

for i in range(4):
    img_path = '../data/img%d.JPG'%i
    img = Image.open(img_path)

    pred, prob = predict(img_path, resnet50_model)
    print('{} - Predicted: {}, Probablility: {}'.format(img_path, pred, prob))

    plt.subplot(2,2,i+1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(pred[1])
