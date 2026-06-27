from __future__ import annotations

from rknnlite.api import RKNNLite  # type: ignore[import-not-found]

from common.config import AppConfig


class RknnLiteStereoBackend:
    def __init__(self, config: AppConfig) -> None:
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(config.model.path) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {config.model.path}")

        core_mask_name = config.model.npu_core_mask.upper()
        core_mask = getattr(RKNNLite, f"NPU_CORE_{core_mask_name}", getattr(RKNNLite, "NPU_CORE_AUTO", 0))
        if self.rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("Failed to init RKNN runtime")

    def infer(self, left: object, right: object) -> object:
        outputs = self.rknn.inference(inputs=[left, right])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN stereo inference returned no outputs")
        return outputs[0]
