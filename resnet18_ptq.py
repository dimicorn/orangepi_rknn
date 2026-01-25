import torch
import torchvision
from rknn.api import RKNN
import inspect


model_name = "resnet18_int8"

model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, f"{model_name}.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["logits"],
)
print(f"ONNX saved to {model_name}.onnx")


IMGNET_MEAN = [123.675, 116.28, 103.53]
IMGNET_STD = [58.395, 57.12, 57.375]

rknn = RKNN()
rknn.config(
    target_platform='rk3588',
    mean_values=[IMGNET_MEAN],
    std_values=[IMGNET_STD],
)

print(inspect.signature(rknn.config))

assert rknn.load_onnx(f'{model_name}.onnx', input_size_list=[[3, 224, 224]]) == 0
assert rknn.build(do_quantization=False) == 0
assert rknn.export_rknn(f'{model_name}.rknn') == 0

rknn.release()
print(f"RKNN saved to {model_name}.rknn")