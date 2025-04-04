from itertools import product as product
from math import ceil

import numpy as np
import torch


class PriorBox:
    def __init__(self, cfg, format: str = "tensor", image_size=None, phase="train"):
        super().__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"
        self.__format = format

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        if self.__format == "tensor":
            output = torch.Tensor(anchors).view(-1, 4)
        elif self.__format == "numpy":
            output = np.array(anchors).reshape(-1, 4)
        else:
            print(TypeError("ERROR: INVALID TYPE OF FORMAT"))

        if self.clip:
            if self.__format == "tensor":
                output.clamp_(max=1, min=0)
            else:
                output = np.clip(output, 0, 1)

        return output
