import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs
    

class unetUp_MRC(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp_MRC, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.wondow_conv = windowConv(in_size, out_size, temp_channel=False)

    def long_kernel_conv(self,input_channel, output_channel, kernel_size=(3,3), padding=1, dilation=1):
        conv = nn.Sequential(torch.nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding, dilation=dilation),
                                    nn.BatchNorm2d(output_channel),
                                    nn.ReLU(inplace = True))
        return conv
    
    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.wondow_conv(outputs)
        return outputs


class zeroConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(zeroConv,self).__init__()   
        # tem_size = int(out_size/2)
        tem_size = 256
        self.conv0 = nn.Sequential(torch.nn.Conv2d(in_size, tem_size, kernel_size=3, padding=1, dilation=1),
                                    nn.BatchNorm2d(tem_size),
                                    nn.ReLU(inplace = True))
        self.conv1 = nn.Sequential(torch.nn.Conv2d(tem_size, tem_size, kernel_size=3, padding=2, dilation=2),
                                    nn.BatchNorm2d(tem_size),
                                    nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(torch.nn.Conv2d(tem_size, tem_size, kernel_size=3,  padding=4, dilation=4),
                                    nn.BatchNorm2d(tem_size),
                                    nn.ReLU(inplace = True))
        self.conv3 = nn.Sequential(torch.nn.Conv2d(tem_size, tem_size, kernel_size=3, padding=8, dilation=8),
                                    nn.BatchNorm2d(tem_size),
                                    nn.ReLU(inplace = True))
        self.conv4 = nn.Sequential(torch.nn.Conv2d(tem_size, tem_size, kernel_size=3, padding=16, dilation=16),
                                    nn.BatchNorm2d(tem_size),
                                    nn.ReLU(inplace = True))

        self.conv5 = nn.Sequential(torch.nn.Conv2d(tem_size*5+in_size, out_size, kernel_size=3, padding=1, dilation=1),
                                    nn.BatchNorm2d(out_size),
                                    nn.ReLU(inplace = True))
       

    def forward(self, inputs):
        outputs_0 = self.conv0(inputs)
        outputs_1 = self.conv1(outputs_0)
        outputs_2 = self.conv2(outputs_1)
        outputs_3 = self.conv3(outputs_2)
        outputs_4 = self.conv4(outputs_3)
        # outputs   = torch.cat([outputs_0, outputs_1, outputs_2, outputs_3, outputs_4], 1)
        outputs   = torch.cat([inputs, outputs_0, outputs_1, outputs_2, outputs_3, outputs_4], 1)
        outputs   = self.conv5 (outputs)

        return outputs

    def long_kernel_conv(self,input_channel, output_channel, kernel_size, padding, dilation):
        conv = nn.Sequential(torch.nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding, dilation=dilation),
                                    nn.BatchNorm2d(output_channel),
                                    nn.ReLU(inplace = True))


###############################################################
class windowConv(nn.Module):
    def __init__(self, input_channel, output_channel, temp_channel=True):
        super(windowConv,self).__init__()   
        self.input_channel = input_channel
        self.output_channel = output_channel
        if temp_channel:
            self.temp_channel = int(self.input_channel/2)
        else:
            self.temp_channel = 32
        self.conv0_1 = self.long_kernel_conv(self.input_channel, self.temp_channel, kernel_size=(3,15), padding=(1,7), dilation=1)
        self.conv0_2 = self.long_kernel_conv(self.input_channel, self.temp_channel, kernel_size=(15,3), padding=(7,1), dilation=1)
        self.conv0_0 = self.long_kernel_conv(self.temp_channel*2, self.temp_channel, kernel_size=(1,1), padding=(0,0), dilation=1)

        self.conv1_1 = self.long_kernel_conv(self.input_channel, self.temp_channel, kernel_size=(3,15), padding=(2,14), dilation=2)
        self.conv1_2 = self.long_kernel_conv(self.input_channel, self.temp_channel, kernel_size=(15,3), padding=(14,2), dilation=2)
        self.conv1_0 = self.long_kernel_conv(self.temp_channel*2, self.temp_channel, kernel_size=(1,1), padding=(0,0), dilation=1)

        self.conv2_1 = self.long_kernel_conv(self.input_channel, self.temp_channel, kernel_size=(3,15), padding=(4,28), dilation=4)
        self.conv2_2 = self.long_kernel_conv(self.input_channel, self.temp_channel, kernel_size=(15,3), padding=(28,4), dilation=4)
        self.conv2_0 = self.long_kernel_conv(self.temp_channel*2, self.temp_channel, kernel_size=(1,1), padding=(0,0), dilation=1)

        self.conv = self.long_kernel_conv(self.temp_channel*3, self.output_channel, kernel_size=(1,1), padding=(0,0), dilation=1)
 

    def long_kernel_conv(self,input_channel, output_channel, kernel_size=(3,3), padding=1, dilation=1):
        conv = nn.Sequential(torch.nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding, dilation=dilation),
                                    nn.BatchNorm2d(output_channel),
                                    nn.ReLU(inplace = True))
        return conv
    

    def forward(self, inputs):
        x_01 = self.conv0_1(inputs)
        x_02 = self.conv0_2(inputs)
        x_00 = torch.cat([x_01, x_02], 1)
        x_00 = self.conv0_0(x_00)

        x_11 = self.conv1_1(inputs)
        x_12 = self.conv1_2(inputs)
        x_10 = torch.cat([x_11, x_12], 1)
        x_10 = self.conv1_0(x_10)

        x_21 = self.conv2_1(inputs)
        x_22 = self.conv2_2(inputs)
        x_20 = torch.cat([x_21, x_22], 1)
        x_20 = self.conv2_0(x_20)

        x = torch.cat([x_00, x_10, x_20], 1)
        x = self.conv(x)

        return x 

###############################################################


class OcclusionNet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', window=False, MD_Module=False, MRC_Module=False):
        super(OcclusionNet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
            self.zero_conv = zeroConv(512, 512)
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
            self.zero_conv = zeroConv(2048, 2048)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        self.MD_Module = MD_Module
        self.MRC_Module = MRC_Module
        ##########################################################


        # upsampling
        if MRC_Module:
            # 64,64,512
            self.up_concat4 = unetUp_MRC(in_filters[3], out_filters[3])
            # 128,128,256
            self.up_concat3 = unetUp_MRC(in_filters[2], out_filters[2])
            # 256,256,128
            self.up_concat2 = unetUp_MRC(in_filters[1], out_filters[1])
            # 512,512,64
            self.up_concat1 = unetUp_MRC(in_filters[0], out_filters[0])
        else:
            # 64,64,512
            self.up_concat4 = unetUp(in_filters[3], out_filters[3])
            # 128,128,256
            self.up_concat3 = unetUp(in_filters[2], out_filters[2])
            # 256,256,128
            self.up_concat2 = unetUp(in_filters[1], out_filters[1])
            # 512,512,64
            self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        # MD_Module
        if self.MD_Module:
            zero_1 = self.zero_conv(feat5)
            up4 = self.up_concat4(feat4, zero_1)
        else:
            up4 = self.up_concat4(feat4, feat5)

        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        ##########################################################
        # windowConv
        # if not self.MRC_Module:
        #     if self.window:
        #         up1 = self.window_conv(up1)
        ##########################################################
        final = self.final(up1)

        return final


    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg": 
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
