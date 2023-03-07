from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
import models.vit as vit
from models.swin import SwinTransformer
from models.t2t import t2t_vit_14 as t2t
import torchvision
import torch
import torch.nn as nn
from collections import OrderedDict



class FPNT2T(nn.Module):
    def __init__(self, num_classes=91, pretrained_path=None, fixed_size=(224,224), backbone_out_chan=512):
        super(FPNT2T, self).__init__()

        model  = t2t(num_classes = num_classes)
        norm_layer = nn.LayerNorm

        self.patch_size = 16
        self.num_patches = (fixed_size[0] // self.patch_size) ** 2
        self.embed_dim =  model.embed_dim
        self.fpn = FeaturePyramidNetwork([self.embed_dim for i in range(4)], backbone_out_chan, extra_blocks=LastLevelMaxPool())
        self.pos_embed =  model.pos_embed

        for i_layer in range(4):
            layer = norm_layer(self.embed_dim)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)['state_dict']
            state_dict = {k:v for k,v in state_dict.items() if 'backbone' in k}
            for key in list(state_dict):
                new_key = key.replace('backbone.','')
                state_dict[new_key] = state_dict.pop(key)
            model.load_state_dict(state_dict)
        self.backbone = model

    def forward(self, x):
        out = self.backbone.forward_features(x)
        del out['z']
        del out['z_patches']


        for en, i in enumerate(out.keys()):
            layer = getattr(self, f'norm{en}')
            x = layer(out[i])
            x = x.reshape(x.shape[0], int(self.num_patches ** 0.5), int(self.num_patches ** 0.5), -1).permute(0,3,1,2).contiguous()
            out[i] = x

        x = self.fpn(OrderedDict(out))
        return x
    

class FPNSwin(nn.Module):
    def __init__(self, num_classes = 10, pretrained_path=None, backbone_out_chan=512, vit_type='tiny'):
        super(FPNSwin, self).__init__()

        if vit_type in ['tiny']:
            depths=[2, 2, 6, 2]
        elif vit_type in ['small']:
            depths=[2, 2, 18, 2]
        else:
            raise ValueError('Vit type not supported. tiny, small are accepted.')

        model = SwinTransformer(num_classes = num_classes, use_positional_embeddings=True, depths=depths)
        self.patch_res = [56, 28, 14, 7]
        self.in_channels = [48, 96, 192, 768]
        self.num_features = [192, 384, 768, 768]
        self.fpn = FeaturePyramidNetwork(self.in_channels, backbone_out_chan, extra_blocks=LastLevelMaxPool())
        norm_layer = nn.LayerNorm

        for i_layer in range(4):
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)['state_dict']
            state_dict = {k:v for k,v in state_dict.items() if 'backbone' in k}
            for key in list(state_dict):
                new_key = key.replace('backbone.','')
                state_dict[new_key] = state_dict.pop(key)
            model.load_state_dict(state_dict)
        self.backbone = model

    def forward(self, x):
        x, out = self.backbone.forward_features(x)

        out = OrderedDict({
             'feat0':out[0],
             'feat1':out[1],
             'feat2':out[2],
             'feat3':out[3],
             })

        for en, i in enumerate(out.values()):

            layer = getattr(self, f'norm{en}')
            x = layer(i)
            x = x.reshape(i.shape[0], self.patch_res[en], self.patch_res[en], -1).permute(0,3,1,2).contiguous()
            out['feat{}'.format(en)] = x

        x = self.fpn(out)
        return x
    
class FPNViT(nn.Module):
    def __init__(self, num_classes=91, vit_type='small', pretrained_path=None, fixed_size=(224,224), backbone_out_chan=512):
        super(FPNViT, self).__init__()

        vit_type = 'vit_' + vit_type
        model  = vit.__dict__[vit_type](num_classes=num_classes, use_clf_token=True, use_positional_embeddings=True)
        norm_layer = nn.LayerNorm

        self.patch_size = 16
        self.num_patches = (fixed_size[0] // self.patch_size) ** 2
        self.embed_dim =  model.embed_dim
        self.fpn = FeaturePyramidNetwork([self.embed_dim for i in range(3)], backbone_out_chan, extra_blocks=LastLevelMaxPool())
        self.patch_embed = model.patch_embed
        self.pos_embed =  model.pos_embed

        for i_layer in range(3):
            layer = norm_layer(self.embed_dim)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)['state_dict']
            state_dict = {k:v for k,v in state_dict.items() if 'backbone' in k}
            for key in list(state_dict):
                new_key = key.replace('backbone.','')
                state_dict[new_key] = state_dict.pop(key)
            model.load_state_dict(state_dict)
        self.backbone = model

    def forward(self, x):
        out = self.backbone(x)
        if 'z' in out.keys():
            del out['z']

        for en, i in enumerate(out.keys()):
            layer = getattr(self, f'norm{en}')
            x = layer(out[i])
            x = x.reshape(x.shape[0], int(self.num_patches ** 0.5), int(self.num_patches ** 0.5), -1).permute(0,3,1,2).contiguous()
            out[i] = x

        x = self.fpn(OrderedDict(out))
        return x
    
def create_model(num_classes = 91, pretrained_path = None, fixed_size=(224,224),
                 mode='segm', vit_type='small', model_type='vit', backbone_out_chan = 512,
                 image_mean = [0.485, 0.456, 0.406], image_std = [0.229, 0.224, 0.225]):

    if model_type in ['swin']:
        backbone = FPNSwin(num_classes = num_classes, pretrained_path = pretrained_path, backbone_out_chan = backbone_out_chan, vit_type=vit_type)
        feat_maps, k = ['feat0','feat1','feat2','feat3','pool'], 5
    elif model_type in ['vit']:
        backbone = FPNViT(num_classes = num_classes, pretrained_path = pretrained_path, vit_type=vit_type, backbone_out_chan = backbone_out_chan)
        feat_maps, k = ['feat0','feat1','feat2','pool'], 4
    elif model_type in ['t2t']:
        backbone = FPNT2T(num_classes=num_classes, pretrained_path = pretrained_path, backbone_out_chan=backbone_out_chan, fixed_size=fixed_size)
        feat_maps, k = ['feat0','feat1','feat2','feat3','pool'], 5
    elif model_type in ['resnet']:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes+1)
        return model
    else:
        raise ValueError(f'Model type {model_type} not correct. vit, swin are accepted.')

    backbone.out_channels = backbone_out_chan
    anchorgen = AnchorGenerator(sizes=tuple([(32,64,128,256) for i in range(k)]),
                                aspect_ratios=tuple([(0.5, 1.0, 2.0) for i in range(k)]),)
    pooler = MultiScaleRoIAlign(featmap_names=feat_maps, output_size=7, sampling_ratio=2)

    if mode in ['segm']:
        mask_roi_pooler = MultiScaleRoIAlign(featmap_names=feat_maps, output_size=14,sampling_ratio=2)
        model = MaskRCNN(backbone = backbone,
                    fixed_size=fixed_size,
                    img_mean = image_mean,
                    img_std = image_std,
                    num_classes = num_classes + 1,
                    rpn_anchor_generator = anchorgen,
                    box_roi_pool = pooler,
                    mask_roi_pool = mask_roi_pooler,
                    box_detections_per_img = 100)
        return model

    elif mode in ['bbox']:
        model = FasterRCNN(backbone = backbone,
                    fixed_size = fixed_size,
                    img_mean = image_mean,
                    img_std = image_std,
                    num_classes = num_classes + 1,
                    rpn_anchor_generator = anchorgen,
                    box_roi_pool = pooler,
                    box_detections_per_img = 100
                    )
        return model

    else:
        raise ValueError(f'mode type {mode} not correct. segm, bbox types are accepted.')