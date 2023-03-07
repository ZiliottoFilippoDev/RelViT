from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from torchvision.ops import masks_to_boxes

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

class Resize(nn.Module):
    def __init__(
        self,
        size: int = 224
    ):
      super().__init__()
      self.height = size
      self.width = size
      self.size = size

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

        orig_w, orig_h  = image.shape[1], image.shape[2]
        image = F.resize(image, size = (self.size,self.size))

        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if len(target['boxes']) == 0:
            return image, target

        else:
          target['masks'] = T.Resize((self.size, self.size))(target['masks'])

          bbox = target['boxes']
          y_scale = float(self.size) / orig_w
          x_scale = float(self.size) / orig_h

          bbox[:, 0] = x_scale * bbox[:, 0]
          bbox[:, 2] = x_scale * bbox[:, 2]
          bbox[:, 1] = y_scale * bbox[:, 1]
          bbox[:, 3] = y_scale * bbox[:, 3]
          target['boxes'] = bbox

          return image, target

class RandomRotation(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        angle: float = 20
    ):
      super().__init__()
      self.prob = p
      self.angle = angle
      self.rotation = T.RandomRotation(self.angle)

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

        if torch.rand(1) > self.prob:
            return image, target

        else:
            image = self.rotation(image)
            target['masks'] = self.rotation(target['masks'])
            target['boxes'] = masks_to_boxes(target['masks'])
            return image, target

class RandomPerspective(nn.Module):
    def __init__(
        self,
        p: float = .2,
        distortion: float = .2
    ):
      super().__init__()
      self.prob = p
      self.distortion = distortion
      self.perspective = T.RandomPerspective(self.distortion)

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

        if torch.rand(1) > self.prob:
            return image, target

        else:
            image = self.perspective(image)
            target['masks'] = self.perspective(target['masks'])
            target['boxes'] = masks_to_boxes(target['masks'])
            return image, target

class RandomTranslate(nn.Module):
    def __init__(
        self,
        p: float = .2,
        translate: float = .2
    ):
      super().__init__()
      self.prob = p
      self.translate = translate
      self.perspective = T.RandomAffine(degrees=0., translate=(0.2,0.1))
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

        if torch.rand(1) > self.prob:
            return image, target

        else:
            image = self.perspective(image)
            target['masks'] = self.perspective(target['masks'])
            target['boxes'] = masks_to_boxes(target['masks'])
            return image, target

class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError