"""Post-processing for SEMANTIC ONNX models (exported by `python -m src.export_onnx`), as a drop-in for
`libs.postprocess.postprocess_yolo_masks` of the Chisel server. Same signature, same return value
(uint8 mask, 0 or 255, in the shape of the tile), so the rest of the server (slicing, thread pool, alpha masking,
winner-take-all in MaskProcessor, colouring, areas) stays unchanged.

The model output is `probs` (1, C, S, S): sigmoid probabilities in the letterboxed S x S input space.
Steps (the same geometry as `process_masks` of the YOLO path): remove the letterbox padding, resize the
probabilities back to the tile size, threshold. `conf_threshold` is the class threshold from the sidecar
<model>.onnx.json (best at TTA "none"). `use_bbox_cropping` is accepted and ignored: there are no boxes.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def postprocess_semantic_masks(
    outputs: List[np.ndarray],
    original_shape: Tuple[int, int],
    ratio: float,
    pad: Tuple[float, float],
    input_size: int = 1280,
    conf_threshold: float = 0.5,
    use_bbox_cropping: bool = False,
) -> Optional[np.ndarray]:
    if not outputs or outputs[0].size == 0:
        return None
    probs = outputs[0][0, 0]                                   # (S, S): first (only) class channel

    dw, dh = pad
    top, left = max(0, int(round(dh - 0.1))), max(0, int(round(dw - 0.1)))
    bottom = min(input_size, top + int(round(original_shape[0] * ratio)))
    right = min(input_size, left + int(round(original_shape[1] * ratio)))

    tile = cv2.resize(probs[top:bottom, left:right], (original_shape[1], original_shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return (tile > conf_threshold).astype(np.uint8) * 255
