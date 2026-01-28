import os
from argparse import ArgumentParser
from enum import Enum
from yaml import safe_load
from munch import munchify
from models.resnet import ResNet18
from models.efficientnet import EfficientNetB0
from models.yolov5 import YOLOv5su
from models.yolov8 import YOLOv8s
from models.fcn import FCN_ResNet50
from models.deeplabv3 import DeepLabV3_ResNet50


class ModelFamily(Enum):
    RESNET = "resnet"
    EFFNET = "effnet"
    YOLOv5 = "yolov5"
    YOLOv8 = "yolov8"
    FCN = "fcn"
    DEEPLABv3 = "deeplabv3"


class QuantType(Enum):
    PTQ = "ptq"
    QAT = "qat"


def main() -> None:
    with open("weights.yaml") as f:
        weights_cfg = munchify(safe_load(f))
    parser = ArgumentParser(description="PyTorch to RKNN model converter")
    parser.add_argument(
        "-m",
        "--model",
        type=ModelFamily,
        choices=list(ModelFamily),
        required=True,
    )
    parser.add_argument(
        "-q",
        "--quantize",
        type=QuantType,
        choices=list(QuantType),
    )
    args = parser.parse_args()

    match args.model:
        case ModelFamily.RESNET:
            NET = ResNet18
            model_name = "resnet18"
            dataset = "ILSVRC2012_val_250.txt"
        case ModelFamily.EFFNET:
            NET = EfficientNetB0
            model_name = "efficientnetb0"
            dataset = "ILSVRC2012_val_250.txt"
        case ModelFamily.YOLOv5:
            NET = YOLOv5su
            model_name = "yolov5su"
            dataset = ...
        case ModelFamily.YOLOv8:
            NET = YOLOv8s
            model_name = "yolov8s"
            dataset = ...
        case ModelFamily.FCN:
            NET = FCN_ResNet50
            model_name = "fcn_resnet50"
            dataset = ...
        case ModelFamily.DEEPLABv3:
            NET = DeepLabV3_ResNet50
            model_name = "deeplabv3_resnet50"
            dataset = ...
    
    model_name = os.path.join(weights_cfg.weights_dir, model_name)

    if args.quantize is None:
        net = NET("_".join([model_name, "fp16"]))
    if args.quantize == QuantType.PTQ:
        net = NET(
            "_".join([model_name, "int8"]),
            True,
            QuantType.PTQ.value,
            dataset,
        )
    else:
        net = NET(
            "_".join([model_name, "int8", QuantType.QAT.value]),
            True,
            QuantType.QAT.value,
            dataset,
        )
    net.to_rknn()


if __name__ == "__main__":
    main()
