import os
from typing import Literal
import torch
from rknn.api import RKNN
import inspect


class Model:
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        out_dir = os.path.dirname(model_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if quantize and quantize_type is None:
            raise TypeError("quantize_type should not be None")
        if quantize and dataset is None:
            raise TypeError("dataset should not be None")
        self.model_name = model_name
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.dataset = dataset
        self.model = None
        self.MEAN: list[float] | None = None
        self.STD: list[float] | None = None
        self.input_size: list[float] | None = None

    def _to_onnx(self) -> None:
        dummy = torch.randn(1, *self.input_size)
        torch.onnx.export(
            self.model,
            dummy,
            f"{self.model_name}.onnx",
            opset_version=12,
            input_names=["input"],
            output_names=["logits"],
        )

    def to_rknn(self, verbose: bool = False, to_inspect: bool = False) -> None:
        if self.quantize_type in ["ptq", None]:
            self._to_onnx()

        rknn = RKNN(verbose=verbose)
        rknn.config(
            target_platform="rk3588",
            float_dtype="float16",
            mean_values=[self.MEAN],
            std_values=[self.STD],
        )
        if to_inspect:
            print(inspect.signature(rknn.config))
        assert rknn.load_onnx(
            f"{self.model_name}.onnx",
            input_size_list=[self.input_size],
        ) == 0
        if self.quantize:
            assert rknn.build(do_quantization=True, dataset=self.dataset) == 0
        else:
            assert rknn.build(do_quantization=False) == 0
        assert rknn.export_rknn(f"{self.model_name}.rknn") == 0
        rknn.release()


class ClassificationModel(Model):
    def __init__(
        self,
        model_name: str,
        quantize: bool = False,
        quantize_type: Literal["ptq", "qat"] | None = None,
        dataset: str | None = None,
    ) -> None:
        super().__init__(model_name, quantize, quantize_type, dataset)
        # ImageNet
        self.MEAN = [123.675, 116.28, 103.53]
        self.STD = [58.395, 57.12, 57.375]
        self.input_size = [3, 224, 224]
        