from models.resnet import ResNet18
from models.efficientnet import EfficientNetB0
from models.yolov5 import YOLOv5su
from models.yolov8 import YOLOv8s
from models.fcn import FCN_ResNet50
from models.deeplabv3 import DeepLabV3_ResNet50


def main():
    # resnet = ResNet18(
    #     "weights/resnet18_int8_qat", True, "qat", "ILSVRC2012_val_250.txt"
    # )
    # resnet.to_rknn()
    net = YOLOv8s(
        "weights/yolov8s_fp16",
        # True,
        # "qat",
        # "ILSVRC2012_val_250.txt",
    )
    net.to_rknn()


if __name__ == "__main__":
    main()
