import collections
import os
from xml.etree.ElementTree import Element as ET_Element
import torch
from torchvision.datasets import VisionDataset
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2011": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "filename": "VOCtrainval_25-May-2011.tar",
        "md5": "6c3384ef61512963050cb5d687e5bf1e",
        "base_dir": os.path.join("TrainVal", "VOCdevkit", "VOC2011"),
    },
    "2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "md5": "da459979d0c395079b5c75ee67908abb",
        "base_dir": os.path.join("VOCdevkit", "VOC2010"),
    },
    "2009": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "filename": "VOCtrainval_11-May-2009.tar",
        "md5": "a3e00b113cfcfebf17e343f59da3caa1",
        "base_dir": os.path.join("VOCdevkit", "VOC2009"),
    },
    "2008": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "2629fa636546599198acfcfbfcf1904a",
        "base_dir": os.path.join("VOCdevkit", "VOC2008"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
    "2007-test": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "b6e924de25625d8de591ea690078ad9f",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}


class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.year = verify_str_arg(year, "year", valid_values=[str(yr) for yr in range(2007, 2013)])

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.targets)
    def __len__(self) -> int:
        return len(self.images)


voc_labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
              'cow','diningtable','dog','horse','motorbike'     ,'person'       ,'pottedplant',
              'sheep'   ,'sofa','train','tvmonitor']
voc_labels = dict(map(reversed, enumerate(voc_labels)))

class VOCDetection(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    @property
    def annotations(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        ann = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())['annotation']['object']

        order = ['xmin','ymin','xmax','ymax']
        boxes = [i['bndbox'] for i in ann]
        boxes = [{k: i[k] for k in order} for i in boxes]
        boxes = [list(map(int, val.values())) for val in boxes]
        boxes = torch.tensor(boxes).to(torch.float64)

        labels = [i['name'] for i in ann]
        labels = [voc_labels[i]+1 for i in labels]
        labels = torch.tensor(labels)


        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]

            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict