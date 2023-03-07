import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from maskvit import create_model
from datasets.coco_utils import get_coco, get_coco_api_from_dataset
from datasets.engine import _get_iou_types
from datasets.coco_eval import CocoEvaluator
from datasets.presets import DetectionPresetEval, DetectionPresetTrain
from datasets.engine import evaluate
from datasets.visualize import display_instances, coco_labels, bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from models.utils import get_pos_embedding_sim
from torchvision.utils import make_grid
from matplotlib import cm
from sklearn.decomposition import PCA
from PIL import Image
cmap = cm.get_cmap('viridis')

def collate_fn(batch):
    return tuple(zip(*batch))

class ObjectDetector(pl.LightningModule):
    def __init__(self, num_classes, lr=0.0001, weight_decay=0.0005, t_max=10, vit_type='small',
                  batch_train = 128, batch_val = 8, num_workers = 2, pretrained_path=None,
                  only_use_val = False, seed = 1234, gamma = 0.1, step_size = 10,
                  fixed_size=224, top_scores = 6, cat_list=None, mode='segm', augmentation='ssd',
                  subset=None, lr_name ='adam', model_type='vit',backbone_out_chan=512):
        super().__init__()

        if cat_list is not None:
            num_classes = len(cat_list) + 1
            self.cat_list = [int(i) for i in cat_list]
        else: self.cat_list = None

        self.seed = seed
        pl.seed_everything(self.seed, workers=True)
        self.model = create_model(num_classes=num_classes, pretrained_path=pretrained_path,
                                     fixed_size=(fixed_size, fixed_size),model_type=model_type,
                                     vit_type=vit_type, mode=mode, backbone_out_chan=backbone_out_chan)
        self.map_box = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',class_metrics=True)
        self.map_mask = MeanAveragePrecision(iou_type='segm', class_metrics=True)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_train = batch_train
        self.batch_val = batch_val
        self.step_size = step_size
        self.gamma = gamma
        self.num_workers = num_workers
        self.only_use_val = only_use_val
        self.top_scores = top_scores
        self.augmentation = augmentation
        self.subset = subset
        self.mode = mode
        self.lr_name = lr_name
        self.model_type = model_type
        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images  = list(image for image in batch[0])
        targets = [{k:v for k, v in t.items()} for t in batch[1]]

        loss_dict = self.model(images, targets)
        self.log_losses(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0:
            images  = list(image for image in batch[0])
            targets = [{k:v for k,v in t.items()} for t in batch[1]]
            preds = self.model(images)
            preds = [{k:v for k,v in t.items()} for t in preds]

            if self.mode == 'segm':
                for i in range(len(preds)):
                    preds[i]['masks'] = preds[i]['masks'] > 0.5
                    preds[i]['masks'] = preds[i]['masks'].to(torch.uint8).squeeze(1)
                for i in range(len(targets)):
                    targets[i]['masks'] = targets[i]['masks'].to(torch.uint8)
                self.map_mask.update(preds=preds, target=targets)
            self.map_box.update(preds=preds, target=targets)
            if batch_idx == 1 or batch_idx == 20:
                self.log_bounding_boxes(images, targets, preds, coco_labels, self.current_epoch)

        if batch_idx == 0 and self.model_type in ['vit']:
            pos_sim_matrix = get_pos_embedding_sim(self.model.backbone.pos_embed.detach().cpu())
            num_patches = pos_sim_matrix.shape[0]
            pos_sim_matrix = pos_sim_matrix.view(-1, num_patches, num_patches).unsqueeze(1)
            pos_sim_matrix = (pos_sim_matrix + 1.0) / 2.0
            pos_sim_matrix = make_grid(pos_sim_matrix, nrow=num_patches, padding=1, pad_value=-1)[0]
            mask = torch.where(pos_sim_matrix >= 0, torch.ones_like(pos_sim_matrix), torch.zeros_like(pos_sim_matrix))
            mask = torch.stack([torch.ones_like(mask), torch.ones_like(mask), torch.ones_like(mask), mask], -1)
            pos_sim_matrix = torch.tensor(cmap(pos_sim_matrix))
            pos_sim_matrix = (pos_sim_matrix * mask).permute(2, 0, 1)
            self.logger.experiment.log({'pos_embed_sim': [wandb.Image(pos_sim_matrix)]})

            kernel = list(self.model.backbone.patch_embed.parameters())[0].detach().cpu().numpy()
            D, C, PS, PS = kernel.shape
            pca = PCA(n_components=28)
            rgb_filt = pca.fit_transform(kernel.reshape(D, -1).T).T.reshape(-1, C, PS, PS)
            rgb_filt = (rgb_filt - rgb_filt.min()) / (rgb_filt.max() - rgb_filt.min())
            rgb_filt = torch.tensor(rgb_filt)
            rgb_filt = make_grid(rgb_filt, nrow=7, padding=1)
            self.logger.experiment.log({'rgb_emb_filters': [wandb.Image(rgb_filt)]})

    def validation_epoch_end(self, validation_step_outputs):
        if self.current_epoch % 10 == 0:
            mAPs_box = {'val_box_' + k:v for k,v in self.map_box.compute().items()}
            maps_per_class = mAPs_box.pop("val_box_map_per_class")
            mars_per_class = mAPs_box.pop('val_box_mar_100_per_class')
            self.log_dict(mAPs_box)
            self.log_dict({'Person mAP box':{f'mAP_box_{label}': values for label, values in zip(coco_labels.values(), maps_per_class)}['mAP_box_person']})
            self.map_box.reset()

            if self.mode == 'segm':
                mAPs_mask = {'val_mask_'+ k:v for k,v in self.map_mask.compute().items()}
                maps_per_class = mAPs_mask.pop("val_mask_map_per_class")
                mars_per_class = mAPs_mask.pop('val_mask_mar_100_per_class')
                self.log_dict(mAPs_mask)
                self.log_dict({'Person mAP mask':{f'mAP_mask_{label}': values for label, values in zip(coco_labels.values(), maps_per_class)}['mAP_mask_person']})
                self.map_mask.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        targets_cpu = []
        outputs_cpu = []
        for target, output in zip(targets, preds):
            t_cpu = {k: v.cpu() for k, v in target.items()}
            o_cpu = {k: v.cpu() for k, v in output.items()}
            targets_cpu.append(t_cpu)
            outputs_cpu.append(o_cpu)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets_cpu, outputs_cpu)
        }
        self.coco_evaluator.update(res)

        if batch_idx == 0:
            images = list(image for image in batch[0])
            targets = [{k:v for k,v in t.items()} for t in batch[1]]
            preds = self.model(images)
            preds = [{k:v for k,v in t.items()} for t in preds]

            if self.mode == 'segm':
                for i in range(len(preds)):
                    preds[i]['masks'] = preds[i]['masks'] > 0.5
                    preds[i]['masks'] = preds[i]['masks'].to(torch.uint8).squeeze(1)
                for i in range(len(targets)):
                    targets[i]['masks'] = targets[i]['masks'].to(torch.uint8)
            self.log_bounding_boxes(images, targets, preds, coco_labels, 'test')
        return res

    def test_epoch_end(self, validation_step_outputs):
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        metric_box = self.coco_evaluator.coco_eval["bbox"]
        metric_segm = self.coco_evaluator.coco_eval["segm"]

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.lr_name in ['adam']:
           optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_name in ['sgd']:
           optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, gamma = self.gamma, step_size = self.step_size)

        return {"optimizer":optimizer, "lr_scheduler": sch}

    def log_losses(self, loss_dict):
        loss_cls = loss_dict['loss_classifier']
        loss_box_reg = loss_dict['loss_box_reg']
        loss_obj = loss_dict['loss_objectness']
        loss_rpn = loss_dict['loss_rpn_box_reg']
        if self.mode == 'segm':
            loss_mask = loss_dict['loss_mask']
        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses, on_step=True, on_epoch=True, batch_size = self.batch_train)
        self.log('train_loss_cls', loss_cls, on_step=True, on_epoch=True, batch_size = self.batch_train)
        self.log('train_loss_box_reg', loss_box_reg, on_step=True, on_epoch=True, batch_size = self.batch_train)
        self.log('train_loss_obj', loss_obj, on_step=True, on_epoch=True, batch_size = self.batch_train)
        self.log('train_loss_rpn', loss_rpn, on_step=True, on_epoch=True, batch_size = self.batch_train)
        if self.mode == 'segm':
            self.log('train_loss_mask', loss_mask, on_step=True, on_epoch=True, batch_size = self.batch_train)

    def log_bounding_boxes(self, images, targets, predictions, class_dict, epoch):
        ids = [i for i in range(len(images))]
        for id, img, preds, targs in zip(ids, images, predictions, targets):
            class_labels_pred = preds['labels'].detach().cpu().tolist()
            class_labels_truth = targs['labels'].detach().cpu().tolist()
            conditions_truth = [i == 1 for i in targs['masks'].detach().cpu()]
            semantic_truth = np.select(conditions_truth, class_labels_truth, default=0)

            if len(preds['labels']) == 0:
                semantic_pred = semantic_truth
            else:
                conditions_pred = [i == 1 for i in preds['masks'].detach().cpu()]
                semantic_pred = np.select(conditions_pred, class_labels_pred, default=0)

            instance_img = wandb.Image(img, boxes = {
             "prediction" : {
                 "box_data" : [{
                    "position" :{
                        "minX" : float(box[0]),
                        "minY" : float(box[1]),
                        "maxX" : float(box[2]),
                        "maxY" : float(box[3])
                    },
                    "class_id" : int(preds['labels'][en]),
                    "box_caption" : "%s (%.2f)" % (class_dict[int(preds['labels'][en])], preds['scores'][en]),
                    "scores" : {'score':float(preds['scores'][en])},
                    "domain" : "pixel"
                }
                for en, box in enumerate(preds['boxes'])
                ],
                "class_labels" : class_dict
            },
             "ground_truth" : {
                "box_data" : [{
                    "position" :{
                        "minX" : float(box[0]),
                        "minY" : float(box[1]),
                        "maxX" : float(box[2]),
                        "maxY" : float(box[3])
                    },
                    "class_id" : int(targs['labels'][en]),
                    "box_caption" : "%s " % (class_dict[int(targs['labels'][en])]),
                    "domain" : "pixel"
                }
                for en, box in enumerate(targs['boxes'])
                ],
                "class_labels" : class_dict
                }
             },
             masks = {
                "predictions": {
                    'mask_data': semantic_pred,
                    'class_labels': class_dict
                },
                 "ground_truth":{
                    'mask_data': semantic_truth,
                    'class_labels': class_dict
                }
               }
            )
            wandb.log({f"Model Output Epoch {epoch}": instance_img})

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
      tform_val = DetectionPresetEval()
      tform_train = DetectionPresetTrain(data_augmentation=self.augmentation)

      self.trainset = get_coco(root='datasets/data', image_set='train', transforms=tform_train, cat_list = self.cat_list)
      self.valset = get_coco(root='datasets/data', image_set='val', transforms = tform_val, cat_list = self.cat_list)
      self.testset = get_coco(root='datasets/data', image_set='val', transforms=tform_val, cat_list = self.cat_list)

      #### SUBSET #####
      if self.subset is not None:
          self.trainset = torch.utils.data.Subset(self.trainset, [i for i in range(self.subset)])
          self.valset = torch.utils.data.Subset(self.valset, [i for i in range(3000)])
          self.testset = torch.utils.data.Subset(self.testset, [i for i in range(3000)])

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.batch_train, num_workers=self.num_workers,
                          collate_fn=collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valset, shuffle=False, batch_size=self.batch_val, num_workers=self.num_workers,
                          collate_fn=collate_fn, pin_memory=True)

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.testset, shuffle=False, batch_size=self.batch_val, num_workers=self.num_workers, collate_fn=collate_fn)
        self.test = test_loader.dataset
        coco = get_coco_api_from_dataset(test_loader.dataset)
        self.iou_types = _get_iou_types(self.model)
        self.coco_evaluator = CocoEvaluator(coco, self.iou_types)
        return test_loader
                          