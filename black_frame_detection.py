import json
import os
import struct
from typing import Sequence

from scannerpy.types import Histogram


def black_frames(config, hists: Sequence[Histogram]) -> Sequence[bytes]:
    output = []
    for h in hists:
        threshold = 0.99 * sum(h[0])
        is_black = (h[0][0] > threshold and h[1][0] > threshold 
                    and h[2][0] > threshold)
        output.append(struct.pack('B', 1 if is_black else 0))

    return output


def get_black_frames_results(video_black_frames):
    return [i for i, b in enumerate(video_black_frames) if ord(b) > 0]
