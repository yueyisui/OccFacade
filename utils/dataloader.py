import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class facadeDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(facadeDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        file_path = os.path.join(os.path.join(self.dataset_path, "Images"), name + ".jpg")
        if os.path.exists(file_path):
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), name + ".jpg"))
        else:
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), name + ".png"))
            
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

# DataLoader中collate_fn使用
def facade_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels


class ECP_Facade_voc_Dataset(Dataset):
    """
    PennFudan 数据集
    'root':路径
    'transforms':数据转换类型
    """
    def __init__(self, root, transforms=None, class_values=[1]):
        self.root = root # 数据集的根路径
        self.transforms = transforms # 数据集的预处理变形参数
        # self.data_test = data_test
        self.class_values = class_values
        
        # 路径组合后返回该路径下的排序过的文件名（排序是为了对齐）
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images")))) # self.imgs 是一个全部待训练图片文件名的有序列表
        self.masks = list(sorted(os.listdir(os.path.join(root, "Labels")))) # self.masks 是一个全部掩码图片文件名的有序列表
 
    # 根据idx对应读取待训练图片以及掩码图片
    def __getitem__(self, idx):
        # 根据idx针对img与mask组合路径
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_path = os.path.join(self.root, "Labels", self.masks[idx])
        
        # 根据路径读取三色图片并转为RGB格式
        img = Image.open(img_path).convert("RGB")
        
        # 根据路径读取掩码图片默认“L”格式
        label = Image.open(mask_path)
        # 将mask转为numpy格式，h*w的矩阵,每个元素是一个颜色id
        label = np.array(label)

        object_masks_info = get_object_masks_info(label, self.class_values)
        boxes = [] # 边界框四个坐标的列表，维度(N,4)
        labels = []
        all_masks=[]
        for j in range(len(self.class_values)):
            mask = object_masks_info[j]['labels']
            num_labels = object_masks_info[j]['num_labels']
            stats = object_masks_info[j]['stats']
            centroids = object_masks_info[j]['centroids']
            id = object_masks_info[j]['id']
            obj_ids = np.unique(mask)
            # 列表中第一个元素代表背景，不属于我们的目标检测范围，obj_ids=[1,2]
            obj_ids = obj_ids[1:]
            # obj_ids[:,None,None]:[[[1]],[[2]]],masks(2,536,559)
            # 为每一种类别序号都生成一个布尔矩阵，标注每个元素是否属于该颜色
            masks = mask == obj_ids[:, None, None]
            # masks = torch.nn.functional.one_hot(mask)
            # 为每个目标计算边界框，存入boxes
            num_objs = len(obj_ids) # 目标个数N
            tem_id = []
            for i in range(num_objs):
                pos = np.where(masks[i]) # pos为mask[i]值为True的地方,也就是属于该颜色类别的id组成的列表
                xmin = np.min(pos[1]) # pos[1]为x坐标，x坐标的最小值
                xmax = np.max(pos[1])
                ymin = np.min(pos[0]) # pos[0]为y坐标
                ymax = np.max(pos[0])
                if xmin<xmax and ymin<ymax:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(id)
                    # tem_masks = masks
                else:
                    tem_id.append(i)
            masks = np.delete(masks, tem_id, axis=0)
            all_masks.extend(masks)
        # 将boxes转化为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 初始化类别标签,目前只有一类，按照顺序标标签
        labels = torch.tensor(labels, dtype=torch.int64)
        all_masks = np.array(all_masks)
        # all_masks = all_masks.reshape(-1, label.shape[0], label.shape[1])
        all_masks = torch.as_tensor(all_masks, dtype=torch.uint8) # 将masks转换为tensor
        # 将图片序号idx转换为tensor
        image_id = torch.tensor([idx])
        # 计算每个边界框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # iscrowd[0,0] (2,)的array

        # 生成一个字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = all_masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 变形transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
 
    def __len__(self):
        return len(self.imgs)


def get_object_masks_info(label_image, class_values):
    '''
    该函数用于实现获取label图像上每个目标对象的mask区域
    input：
        'label_image'：原始的label图像
    return：
        'object_masks_info'：列表，用于存放返回连通区域信息
    '''
    object_masks_info = []  # 初始化
    [h, w] = label_image.shape
    for id in class_values:
        masks = np.zeros((h, w), np.uint8)
        index = label_image == id
        masks[index] = 255
        # 阈值分割得到二值化图片
        ret, th = cv2.threshold(masks,1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 腐蚀与膨胀
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.erode(th,kernel2,iterations = 1)
        th = cv2.dilate(th, kernel2, iterations=1)
        # 连通域提取，并返回每个连通区域的信息（第一个是背景）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8, ltype=None)
        mask_info = {}
        mask_info['num_labels'] = num_labels  # 所有连通域的数目
        mask_info['labels'] = labels  # 图像上每一像素的标记，用数字1、2、3…表示，不同数字表示不同的连通区域
        mask_info['stats'] = stats  # 每个连通区域的外接矩形的x、y、width、height和面积，shape:5列的矩阵
        mask_info['centroids'] = centroids  # 连通域的中心点
        mask_info['id'] = id
        object_masks_info.append(mask_info)
    return object_masks_info