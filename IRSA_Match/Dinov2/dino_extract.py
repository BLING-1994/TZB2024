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

"""Wrapper for performing DINOv2 inference."""

import cv2
import numpy as np
from third_party.dinov2 import dino
import torch
from torchvision.transforms import Resize
import torch.nn as nn

class DINOExtract(nn.Module):
  """Class to initialize DINO model and extract features from an image."""

  def __init__(self, cpt_path: str, feature_layer: int = 1, use_half=True, device = torch.device("cuda"), fixsize=490, image_size_max=1400):
    super(DINOExtract, self).__init__()
    self.fixsize = fixsize
    self.image_size_max = image_size_max
    self.device = device
    self.feature_layer = feature_layer
    if 'vits' in cpt_path:
      self.model = dino.vit_small()
    elif 'vitb' in cpt_path:
      self.model = dino.vit_base()
    state_dict_raw = torch.load(cpt_path, map_location='cpu')

    self.h_down_rate = self.model.patch_embed.patch_size[0]
    self.w_down_rate = self.model.patch_embed.patch_size[1]

    self.model.load_state_dict(state_dict_raw)
    self.model = self.model.to(self.device)
    if use_half:
      self.model.half()
    self.model.eval()
    self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(self.device)
    self.resize = Resize((490, 490))
    self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(self.device)



  def __call__(self, image) -> np.ndarray:
    return self.forward(image)

  def forward(self, image: torch.Tensor):
    """Feeds image through DINO ViT model to extract features.

    Args:
      image: (H, W, 3) numpy array, decoded image bytes, value range [0, 255].

    Returns:
      features: (H // 14, W // 14, C) numpy array image features.
    """

    image_processed = self._process_image(image)
    # image_processed = image_processed.unsqueeze(0).float().to(self.device)
    features = self.model.get_intermediate_layers(image_processed, n=self.feature_layer)[0]
    mean_features = torch.mean(features, dim=(1))
    # features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return mean_features
  
  def _resize_input_image(self, image):
    if self.fixsize!=None:
      image = self.resize(image)
      return image
    """Resizes image such that both dimensions are divisble by down_rate."""
    h_image, w_image = image.shape[2:]
    h_larger_flag = h_image > w_image
    large_side_image = max(h_image, w_image)

    # resize the image with the largest side length smaller than a threshold
    # to accelerate ViT backbone inference (which has quadratic complexity).
    if large_side_image > self.image_size_max:
      if h_larger_flag:
        h_image_target = self.image_size_max
        w_image_target = int(self.image_size_max * w_image / h_image)
      else:
        w_image_target = self.image_size_max
        h_image_target = int(self.image_size_max * h_image / w_image)
    else:
      h_image_target = h_image
      w_image_target = w_image

    h, w = (
        h_image_target // self.h_down_rate,
        w_image_target // self.w_down_rate,
    )
    h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate
    self.resize = Resize((h_resize, w_resize))
    image = self.resize(image)
    return image

  def _process_image(self, image: torch.Tensor) -> torch.Tensor:
    """Turn image into pytorch tensor and normalize it."""
    image = self._resize_input_image(image)

    for i in range(image.shape[0]):
      image[i] = (image[i] - self.mean) / self.std
    return image


