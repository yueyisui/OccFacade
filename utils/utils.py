from re import L
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os 
from matplotlib import pyplot as plt
import scipy.signal
import shutil
from sklearn.model_selection import KFold
import copy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)
    if nw==0:
        nw = 1
    if nh==0:
        nh = 1

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    

def get_miou_png(image, model, args):
    image       = cvtColor(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    image_data, nw, nh  = resize_image(image, (args.input_shape_w, args.input_shape_h))
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()

        if args.model == 'fcn' or args.model == 'deeplab':
            pr = model(images)['out'][0]
        elif args.model == 'Unet' or args.model == 'zero_Unet':
            with torch.no_grad():
                pr = model(images)[0]

        pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr[int((args.input_shape_h - nh) // 2) : int((args.input_shape_h- nh) // 2 + nh), \
                int((args.input_shape_w - nw) // 2) : int((args.input_shape_w - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

    image = Image.fromarray(np.uint8(pr))
    return image

                
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image = image/255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50'  : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


def get_class_info(name):

    if name == './configs/ECP.txt':
        building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255, 255,128,0, 0,0,255, 128,255,255, 0,255,0, 128,128,128])
        name_classes = ["window", "wall", "balcony", "door", "roof", "sky", "shop","chimney"]
    if name =='./configs/ECP_centerNet.txt':
        building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255, 255,128,0, 0,0,255, 128,255,255, 0,255,0, 128,128,128])
        name_classes = ["window", "wall", "balcony", "door", "roof", "sky", "shop","chimney"]
    elif name == './configs/eTRIMS.txt':
        building_color = list([0,0,0, 128,0,0, 128,0,128, 128,128,0, 128,128,128, 128,64,0, 0,128,128, 0,128,0, 0,0,128])
        name_classes = ["building", "car", "door", "pavement", "road", "sky", "vegetation", "window"]
    elif name == './configs/ENPC2014.txt':
        building_color = list([0,0,0, 255,128,0, 0,255,0, 128,0,255, 255,0,0, 255,255,0, 128,255,255, 0,0,255])
        name_classes = ["door", "shop", "balcony", "window", "wall", "sky", "roof"]
    elif name == './configs/RueMonge2014.txt':
        building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255, 255,128,0, 0,0,255, 128,255,255, 0,255,0])
        name_classes = ['Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Sky', 'Shop']
    elif name == './configs/Mesh.txt':
        # building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255])
        # name_classes = ['Wall', 'Roof', 'Window']
        building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255, 255,128,0, 0,0,255, 128,255,255, 0,255,0])
        name_classes = ['Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Sky', 'Shop']
    elif name == './configs/MeshFacade.txt':
        # building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255])
        # name_classes = ['Wall', 'Roof', 'Window']
        # building_color = list([0,0,0, 255,255,0, 255,0,0])
        building_color = list([255,255,0, 255,0,0])
        name_classes = ['wall','Window']
    elif name == './configs/AllMeshFacade.txt':
        building_color = list([255,255,0, 255,0,0])
        name_classes = ['wall','Window']
    else:
        # building_color = list([0,0,0, 255,0,0, 255,255,0, 128,0,255, 255,128,0, 0,0,255, 128,255,255, 0,255,0, 128,128,128])
        building_color = list([255,255,0, 255,0,0])
        # name_classes = ["window", "wall", "balcony", "door", "roof", "sky", "shop","chimney"]
        name_classes = ["window", "wall"]
    return building_color, name_classes
# [255,0,0, 255,255,0, 128,0,255, 255,128,0, 0,0,255, 128,255,255, 0,255,0]

class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(target, n_classes):
        n, h, w = target.size()
        # n_classes = torch.tensor(n_classes).to(device=device)
        one_hot = torch.zeros(n, n_classes, h, w).cuda().scatter_(1, target.view(n, 1, h, w), 1)
        a = one_hot
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1-loss.mean()


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        # os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")



##############################################
# centerNet的loss
##############################################

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def Kfold_get_images_labels(root, new_data_dir, K=5):
    image_dir = os.path.join(root, 'Images')
    label_dir = os.path.join(root, 'Labels')
    image_names = os.listdir(image_dir)
    length = len(image_names)
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    i = 0
    for train_index, test_index in kf.split(image_names):
        i = i+1
        fold_dir = os.path.join(new_data_dir,f'fold{i}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        # fold_dir = os.path.join(new_data_dir,f'fold{i}')
        print(f"fold={i}")
        train_image_file = open(os.path.join(fold_dir, f'fold{i}_train_images.txt'), mode='w')
        train_label_file = open(os.path.join(fold_dir, f'train_fold{i}.txt'), mode='w')
        test_image_file = open(os.path.join(fold_dir, f'fold{i}_test_images.txt'), mode='w')
        test_label_file = open(os.path.join(fold_dir, f'val_fold{i}.txt'), mode='w')
        # train
        for each in train_index:
            if os.path.exists(os.path.join(fold_dir,'train/Images')):
                shutil.copy(os.path.join(image_dir,image_names[each]), os.path.join(fold_dir, 'train/Images'))
            else:
                os.makedirs(os.path.join(fold_dir,'train/Images'))
                shutil.copy(os.path.join(image_dir, image_names[each]), os.path.join(fold_dir, 'train/Images'))
            train_image_file.write(image_names[each][:-4])
            train_image_file.write('\n')
            if os.path.exists(os.path.join(fold_dir,'train/Labels')):
                shutil.copy(os.path.join(label_dir,image_names[each]), os.path.join(fold_dir, 'train/Labels'))
            else:
                os.makedirs(os.path.join(fold_dir,'train/Labels'))
                shutil.copy(os.path.join(label_dir, image_names[each]), os.path.join(fold_dir, 'train/Labels'))
            train_label_file.write(image_names[each][:-4])
            train_label_file.write('\n')
        train_image_file.close()
        train_label_file.close()
        # test
        for each in test_index:
            if os.path.exists(os.path.join(fold_dir,'test/Images')):
                shutil.copy(os.path.join(image_dir,image_names[each]), os.path.join(fold_dir, 'test/Images'))
            else:
                os.makedirs(os.path.join(fold_dir,'test/Images'))
                shutil.copy(os.path.join(image_dir, image_names[each]), os.path.join(fold_dir, 'test/Images'))
            test_image_file.write(image_names[each][:-4])
            test_image_file.write('\n')
            if os.path.exists(os.path.join(fold_dir,'test/Labels')):
                shutil.copy(os.path.join(label_dir,image_names[each]), os.path.join(fold_dir, 'test/Labels'))
            else:
                os.makedirs(os.path.join(fold_dir,'test/Labels'))
                shutil.copy(os.path.join(label_dir, image_names[each]), os.path.join(fold_dir, 'test/Labels'))
            test_label_file.write(image_names[each][:-4])
            test_label_file.write('\n')
        test_image_file.close()
        test_label_file.close()

def copyImage(root, out, file):
    with open(file, "r") as file:
        image_filenames = file.read().splitlines()
    for filename in image_filenames:
        src_path = os.path.join(root, filename + ".jpg")  # 这里假设图像文件扩展名是 .jpg
        dest_path = os.path.join(out, filename + ".jpg")
        try:
            shutil.copy(src_path, dest_path)
            print(f"Copy {filename} to {out}")
        except FileNotFoundError:
            print(f"{filename} not found in {root}")

    print("Done")


def returnCAM(feature_conv, model, class_idx):
    net_name = []
    params = []
    # model_cpu = model.cpu()
    for name, param in model.named_parameters():
        net_name.append(name)
        params.append(param)
    weight_softmax = np.squeeze(params[-2])
    cam = weight_softmax[class_idx,:].squeeze().unsqueeze(1).unsqueeze(2)
    cam = (cam * feature_conv[0].squeeze(0)).sum(0)
    cam_img = ((cam - cam.min()) / (cam.max() - cam.min())).cpu().data.numpy()
    cam_img = np.uint8(255 * cam_img)
    output_cam = []
    size_upsample = (256, 256)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    # cam = _normalize(F.relu(cam, inplace=True)).cpu()
    # mask = to_pil_image(cam.detach().numpy(), mode='F')
    return output_cam

if __name__ == '__main__':
    root = '/data/MeshFacade/Images'
    new_data_dir = '/data/MeshFacade/new2/train/img'
    file = '/data/MeshFacade/new/fold1/train_fold1.txt'
    # Kfold_get_images_labels(root, new_data_dir, K=5)
    copyImage(root, new_data_dir, file)