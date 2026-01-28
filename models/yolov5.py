from typing import Literal
from ultralytics import YOLO
from .model import DetectionModel


class YOLOv5su(DetectionModel):
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        super().__init__(model_name, quantize, quantize_type, dataset)
        yolo = YOLO("weights/yolov5su.pt")
        self.model = yolo.model
        self.model.eval()
