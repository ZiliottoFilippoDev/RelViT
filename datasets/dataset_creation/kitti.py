import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset
import torch

kitti_labels = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'DontCare']
kitti_labels = dict(map(reversed, enumerate(kitti_labels)))

np.random.seed(12345)
validation_idx_ = [np.random.randint(1,7480) for i in range(1000)]
valid_idx_ = [i for i in range(7480)]
train_idx_ = [i for i in valid_idx_ if i not in validation_idx_]

class Kitti(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(
        self,
        root: str = '/datasets/data',
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        subsplit:str = "train",
        download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "testing"
        self.subsplit = subsplit
        self.validation_idx = validation_idx_
        self.train_idx = train_idx_
        self.valid_idx = valid_idx_

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.subsplit == 'train':
            index = self.train_idx[index]
        else:
            index = self.validation_idx[index]

        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        image = Image.open(self.images[index])
        target = self._parse_target(index) if self.train else None
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


    def _parse_target(self, index: int) -> List:
        list_ = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                list_.append(
                    {
                        "type": line[0],
                        # "truncated": float(line[1]),
                        # "occluded": int(line[2]),
                        # "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        # "dimensions": [float(x) for x in line[8:11]],
                        # "location": [float(x) for x in line[11:14]],
                        # "rotation_y": float(line[14]),
                    }
                )

        target = {}

        label = [i['type'] for i in list_]
        label = [w.replace('Misc', 'DontCare') for w in label]
        label = [kitti_labels[i]+1 for i in label]
        target['labels'] = torch.tensor(label).to(torch.int64)
        target['boxes'] = torch.tensor([i['bbox'] for i in list_]).to(torch.float64)
        return target

    def __len__(self) -> int:
        if self.subsplit == 'train':
            return len(self.train_idx)
        else:
            return len(self.validation_idx)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")
    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self._raw_folder, exist_ok=True)

        # download files
        for fname in self.resources:
            download_and_extract_archive(
                url=f"{self.data_url}{fname}",
                download_root=self._raw_folder,
                filename=fname,
            )                   
