# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for performing OmniGlue inference, plus (optionally) SP/DINO."""

import numpy as np
from Dinov2 import dino_extract

import torch
DINO_FEATURE_DIM = 768
MATCH_THRESHOLD = 1e-3

class Dinofeature:
    # TODO: Add class docstring

    def __init__(self, dino_export: str | None = None) -> None:
        """Initialize OmniGlue with specified exports."""
        if torch.cuda.is_available():
            gpu_idx = [i for i in range(torch.cuda.device_count())]
            torch_device = torch.device(f"cuda:{gpu_idx[0]}") # Use the first GPU
        else:
            torch_device = torch.device("cpu")
        
        if dino_export is not None:
            self.dino_extract = dino_extract.DINOExtract(dino_export, feature_layer=1, device=torch_device)

    def Getfeature(self, image: np.ndarray):
        """Find matches between two images using SP and DINO features."""
        return self.dino_extract(image)

   
