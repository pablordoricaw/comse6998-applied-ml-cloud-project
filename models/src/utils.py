import os
import random
import json
import torchvision.transforms as transforms
import torch
import numpy as np
import time
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image
 
cudnn.benchmark = True

with open("../data/test_images/imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)

d = {str(k): [int(k), v[1]] for k, v in class_idx.items()}

def rn50_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

# decode the results into ([predicted class, description], probability)
def predict(img_path, model):
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)

def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

def get_imagenette_dataloader(
    num_samples: int,
    imagenette_root: str = "../data/imagenette_calibration/imagenette2",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Build a DataLoader over the `imagenette2/val` directory for INT8 calibration.
    """
    # 1) Point at the 'val' subfolder
    val_dir = os.path.join(imagenette_root, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"{val_dir} not found – did you extract imagenette2.tgz?")

    # 2) Standard ResNet-50 preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # (optional) normalize if your model was trained with these stats:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    # 3) ImageFolder will auto-label, but we only care about the images
    dataset = ImageFolder(val_dir, transform=transform)

    # 4) Sample exactly `num_samples` out of the full validation set
    total = len(dataset)
    num_samples = min(num_samples, total)
    indices = list(range(total))
    if shuffle:
        random.shuffle(indices)
    indices = indices[:num_samples]
    sampler = SubsetRandomSampler(indices)

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
 
def build_engine(onnx_file_path):
    # logger to capture errors, warnings, and other information during the build and inference phases
    TRT_LOGGER = trt.Logger()

    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    config = builder.create_builder_config()
    # config is where the optimizations are specified
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
    # we have only one image in batch
    #builder.max_batch_size = 1
    # use FP16 mode if possible
    #if builder.platform_has_fast_fp16:
    #    builder.fp16_mode = True

# generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_serialized_network(network, config)
    #context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine

def export_engine(model_name):
    ONNX_FILE_PATH = f"../models/{model_name}.onnx"

    engine = build_engine(ONNX_FILE_PATH)

    with open(f"../models/{model_name}.engine", "wb") as f:
        f.write(engine)

def pt_to_onnx(model_name):
    print("convert from pt to onnx")
    #model_name = "resnet50_base"

    device = torch.device("cuda")
    resnet50_model = torch.load(f"../models/{model_name}.pt", weights_only=False).to(device)
    resnet50_model.eval()

    ONNX_FILE_PATH = f"../models/{model_name}.onnx"
    inputs = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(resnet50_model, inputs, ONNX_FILE_PATH, export_params=True)

    # check model conversion
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)
