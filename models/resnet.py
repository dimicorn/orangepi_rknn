from typing import Literal
from torchvision import models
from .model import ClassificationModel


class ResNet18(ClassificationModel):
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        super().__init__(model_name, quantize, quantize_type, dataset)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
