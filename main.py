from models.resnet import ResNet18
from models.efficientnet import EfficientNetB0


def main():
    # resnet = ResNet18(
    #     "weights/resnet18_int8_qat", True, "qat", "ILSVRC2012_val_250.txt"
    # )
    # resnet.to_rknn()
    net = EfficientNetB0(
        "weights/efficientnetb0_int8_qat",
        True,
        "qat",
        "ILSVRC2012_val_250.txt",
    )
    net.to_rknn()


if __name__ == "__main__":
    main()
