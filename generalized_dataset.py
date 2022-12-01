import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torchvision import transforms
#from compose import RandomHorizontalFlip, RandomRotate, FixedResize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomSizedBBoxSafeCrop(width=224, height=224, erosion_rate = 0.),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.5, rotate_limit=15, shift_limit=0.0625, scale_limit=0.05),
    ToTensorV2(), 
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

transform_val = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomSizedBBoxSafeCrop(width=224, height=224, erosion_rate = 0.),
    ToTensorV2(), 
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


class GeneralizedDataset:
    """
    Main class for Generalized Dataset.
    """
    
    def __init__(self, max_workers=2, verbose=False, mode='train'):
        self.max_workers = max_workers
        self.verbose = verbose
        self.mode = mode
            
    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)
        target = self.get_target(img_id) if self.train else {}

        if self.mode == 'train':
            sample = (image, target)
            #sample = RandomHorizontalFlip(flip_prob=.5)(sample)
            transformed = transform_train(image=image.permute(1,2,0).numpy(), bboxes=target['boxes'],
                                    masks=[i.numpy() for i in target['masks']],
                                    class_labels=target['labels'])
        if self.mode == 'val':
            transformed = transform_val(image=image.permute(1,2,0).numpy(), bboxes=target['boxes'],
                                    masks=[i.numpy() for i in target['masks']],
                                    class_labels=target['labels'])

        image = transformed['image']
        #target['boxes'], target['masks'], target['labels'] = transformed['bboxes'], transformed['masks'], transformed['class_labels']
        target['boxes'] = torch.stack([torch.Tensor(i) for i in transformed['bboxes']])
        target['masks'] = torch.stack([i for i in transformed['masks']])
        target['labels'] = torch.stack([torch.Tensor(i) for i in transformed['class_labels']])

        sample = image, target
        return sample
    
    def __len__(self):
        return len(self.ids)
    
    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return
        
        since = time.time()
        print("Checking the dataset...")
        
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))
        
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
         
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out