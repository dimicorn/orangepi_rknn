from typing import Literal
from torchvision import models
from .model import ClassificationModel


class EfficientNetB0(ClassificationModel):
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        super().__init__(model_name, quantize, quantize_type, dataset)
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.eval()
