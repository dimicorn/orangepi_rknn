from typing import Literal
from torchvision.models import segmentation
from .model import SegmentationModel


class FCN_ResNet50(SegmentationModel):
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        super().__init__(model_name, quantize, quantize_type, dataset)
        self.model = segmentation.fcn_resnet50(
            weights=segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self.model.eval()
