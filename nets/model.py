import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn
# from nets.unet import Unet
from nets.OccFacade import OcclusionNet as OcclusionNet
# from unet import Unet
import torch
import numpy as np

from utils.loss import focal_loss, l1_loss
      
def get_resnet_MaskRCNN(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
 
    return model



def get_model(args, pretrained=True):
    if args.model == 'fcn':
        fcn_resnet50 = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)
        num = fcn_resnet50.classifier[4].in_channels
        fcn_resnet50.classifier[4] = nn.Conv2d(num, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        num = fcn_resnet50.aux_classifier[4].in_channels
        fcn_resnet50.aux_classifier[4] = nn.Conv2d(num, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        # print(fcn_resnet50)
        return fcn_resnet50

    elif args.model == 'deeplab':
        deeplab = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        num = deeplab.classifier[4].in_channels
        deeplab.classifier[4] = nn.Conv2d(num, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        num = deeplab.aux_classifier[4].in_channels
        deeplab.aux_classifier[4] = nn.Conv2d(num, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        # print(deeplab)
        return deeplab


    elif args.model == 'zero_Unet':
        net = OcclusionNet(num_classes=args.num_classes, pretrained=False, backbone=args.backbone, MD_Module=args.MD_Module, MRC_Module=args.MRC_Module)
        print('model_name=zero_Unet')
        if args.backbone=='resnet50':
            print('backbone=resnet50')
            print('load the model_path of unet_resnet_voc...')
            model_path = './model_data/unet_resnet_voc.pth'
        elif args.backbone=='vgg':
            print('backbone=vgg')
            print('load the model_path of unet_vgg_voc...')
            model_path = './model_data/unet_vgg_voc.pth'
        set_model(net, model_path)

    else:
        print(args.model, "is not right")

    return net
    

def set_model(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

