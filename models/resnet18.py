from typing import Literal
import torch
from torchvision import models
from rknn.api import RKNN
import inspect


class ResNet18:
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        if quantize and quantize_type is None:
            raise TypeError("quantize_type should not be None")
        if quantize and dataset is None:
            raise TypeError("dataset should not be None")
        self.model_name = model_name
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.dataset = dataset
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
    
    def _to_onnx(self) -> None:
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self.model,
            dummy,
            f"{self.model_name}.onnx",
            opset_version=12,
            input_names=["input"],
            output_names=["logits"],
        )

    def to_rknn(self, verbose: bool = False, to_inspect: bool = False) -> None:
        if self.quantize_type in ["qat", None]:
            self._to_onnx()
        IMGNET_MEAN = [123.675, 116.28, 103.53]
        IMGNET_STD = [58.395, 57.12, 57.375]

        rknn = RKNN(verbose=verbose)
        rknn.config(
            target_platform="rk3588",
            float_dtype="float16",
            mean_values=[IMGNET_MEAN],
            std_values=[IMGNET_STD],
        )
        if to_inspect:
            print(inspect.signature(rknn.config))
        rknn.load_onnx(
            f"{self.model_name}.onnx",
            input_size_list=[[3, 224, 224]],
        )
        if self.quantize:
            rknn.build(do_quantization=True, dataset=self.dataset)
        else:
            rknn.build(do_quantization=False)
        rknn.export_rknn(f"{self.model_name}.rknn")
        rknn.release()
