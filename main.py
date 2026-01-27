from models.resnet18 import ResNet18


def main():
    resnet = ResNet18("weights/resnet18_fp16")
    resnet.to_rknn()


if __name__ == "__main__":
    main()
