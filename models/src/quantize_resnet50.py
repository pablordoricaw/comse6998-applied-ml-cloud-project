from utils import get_imagenette_dataloader 
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def clean_quantized_model(input_onnx_path, output_onnx_path):
    # Load the ONNX model
    graph = gs.import_onnx(onnx.load(input_onnx_path))

    # Create a new list of nodes
    new_nodes = []
    
    for node in graph.nodes:
        if node.op == "QLinearAdd":
            print(f"⚡ Found QLinearAdd node: {node.name} -> replacing with Add")

            # Extract inputs
            # QLinearAdd has 8 inputs:
            #   a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
            # We only want the real tensors: a and b

            input_a = node.inputs[0]
            input_b = node.inputs[3]

            # Insert normal Add op
            add_node = gs.Node(
                op="Add",
                inputs=[input_a, input_b],
                outputs=node.outputs
            )
            new_nodes.append(add_node)

        elif node.op == "QLinearGlobalAveragePool":
            print(f"⚡ Found QLinearGlobalAveragePool node: {node.name} -> replacing with GlobalAveragePool")

            input_x = node.inputs[0]
            gap_node = gs.Node(
                op="GlobalAveragePool",
                inputs=[input_x],
                outputs=node.outputs
            )
            new_nodes.append(gap_node)

        else:
            # Keep all other nodes
            new_nodes.append(node)

    # Update the graph
    graph.nodes = new_nodes
    graph.cleanup()

    # Export the new ONNX model
    onnx.save(gs.export_onnx(graph), output_onnx_path)
    print(f"✅ Cleaned model saved to {output_onnx_path}")

# --- CalibrationDataReader class for ONNX Runtime ---
class ImagenetteDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = self._create_enum()

        return next(self.enum_data, None)

    def _create_enum(self):
        for data, _ in self.dataloader:
            # ORT expects input to be a dictionary of {input_name: tensor}
            # Here input name is "input", typical for ResNet50
            yield {"input.1": data.numpy()}

    def rewind(self):
        self.enum_data = None
        self.dataloader = iter(self.dataloader)

# --- Prepare your calibration data ---
calib_loader = get_imagenette_dataloader(
    num_samples=512,   # number of calibration images
    batch_size=1
)

your_reader = ImagenetteDataReader(calib_loader)

# --- Now quantize! ---
quantize_static(
    model_input="../models/resnet50_base.onnx",
    model_output="resnet50_qop.onnx",
    calibration_data_reader=your_reader,
    quant_format=QuantFormat.QOperator,    # Insert QuantizeLinear/DequantizeLinear
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,

)

# Run the fixer
clean_quantized_model(
    input_onnx_path="resnet50_qop.onnx",   # Your problematic quantized model
    output_onnx_path="resnet50_trt_cleaned.onnx"  # New fixed model
)

