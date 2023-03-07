from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from detector import ObjectDetector
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import wandb
import argparse
import os
import subprocess
import logging
import sys
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["WANDB_API_KEY"] = '*************************'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--stepsize_lr', type=int, default=10)
parser.add_argument('--gamma_lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--version', type=str, default='baseline')
parser.add_argument('--pl_dir', type=str, default='Pl_checkpoints')
parser.add_argument('--batch_size_train', type=int, default=128)
parser.add_argument('--batch_size_val', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--name', type=str, default='FasterViT')
parser.add_argument('--only_use_val', type=bool, default=False)
parser.add_argument('--seed', type=int, default= 1234)
parser.add_argument('--vit_type', type=str, default='small')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--img_resize', type=int, default= 512)
parser.add_argument('--top_scores', type=int, default=6)
parser.add_argument('--num_gpus',type=int, default=1)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--save_checkpoint_path', type=str, default='/tmp')
parser.add_argument('--cat_list', type=list, default=None)
parser.add_argument('--mode',type=str,default='segm')
parser.add_argument('--augmentation', type=str, default='ssd')
parser.add_argument('--subset',type=int,default=None)
parser.add_argument('--lr_name',type=str,default='adam')
parser.add_argument('--resume_training',type=bool,default=False)
parser.add_argument('--backbone_out_chan',type=int,default=512)
parser.add_argument('--model_type',type=str,default='vit')
args = parser.parse_args()

# Logging
if args.num_gpus > 0:
    nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE).communicate()[0].decode('utf8')
    logging.info(nvidia_smi)
else:
    logging.info('Using CPU!')
    hostname = os.uname().nodename # Node name
    logging.info(f'hostname:{hostname}')

#Example pretrained_path
#~/relvit_coco/tmp/relvit/2f2dijmi/checkpoints/best.ckpt
version = args.version+'/FPN'+args.model_type+'/'+args.vit_type+'/'+args.augmentation+'/'+str(args.img_resize)+'-16'+'/'+str(args.subset)+'k'

lr_monitor = LearningRateMonitor(logging_interval='step')
detector = ObjectDetector(num_classes=91,
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          step_size=args.stepsize_lr,
                          gamma=args.gamma_lr,
                          batch_train = args.batch_size_train,
                          batch_val = args.batch_size_val,
                          num_workers = args.num_workers,
                          only_use_val = args.only_use_val,
                          seed = args.seed,
                          vit_type = args.vit_type,
                          fixed_size = args.img_resize,
                          top_scores = args.top_scores,
                          pretrained_path = args.pretrained_path,
                          cat_list = args.cat_list,
                          mode=args.mode,
                          augmentation=args.augmentation,
                          subset=args.subset,
                          lr_name = args.lr_name,
                          model_type=args.model_type,
                          backbone_out_chan=args.backbone_out_chan
                          )

wandb.init(project=args.name, name=version)
wandb_logger = WandbLogger(project=args.name, name=version)
checkpoint_callback = ModelCheckpoint(dirpath=args.pl_dir,
                                      filename= args.model_type+'_last')

if args.resume_training:
    resume_training_ = args.pl_dir + '/'+ args.model_type+'_last.ckpt'
else:
    resume_training_ = None

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=args.epochs,
    default_root_dir=args.pl_dir,
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoint_callback],
    #deterministic=True,
    )

trainer.fit(detector)
#trainer.test(ckpt_path=args.pl_dir+'/'+args.model_type+"_last.ckpt")
trainer.test(detector)
wandb.finish()
