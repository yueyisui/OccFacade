'''
Description: 
Author: Yue Yisui
Date: 2022-12-07 21:03:21
LastEditTime: 2023-03-20 21:56:27
LastEditors: Yue Yisui
'''
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn

def focal_loss(pred, target):
    """
    classifier loss of focal loss
    Args:
        pred: heatmap of prediction
        target: heatmap of ground truth

    Returns: cls loss

    """
    # Find every image positive points and negative points,
    # one bounding box corresponds to one positive point,
    # except positive points, other feature points are negative sample.
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    # The negative samples near the positive sample feature point have smaller weights
    neg_weights = torch.pow(1 - target, 4)
    loss = 0
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    # Calculate Focal Loss.
    # The hard to classify sample weight is large, easy to classify sample weight is small.
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights

    # Loss normalization is carried out
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def l1_loss(pred, target, mask):
    """
    Calculate l1 loss
    Args:
        pred: offset detection result
        target: offset ground truth
        mask: offset mask, only center point is 1, other place is 0

    Returns: l1 loss

    """
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    # Don't calculate loss in the position without ground truth.
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')

    loss = loss / (mask.sum() + 1e-7)

    return loss



def loss_function(pred,label):
    '''
    简化原理计算整体var的损失函数
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题()
    loss = criterion(pred, label)
    pred = pred.argmax(dim=1)
    pred_numpy = pred.cpu().numpy()
    class_values = [1]  # 1是windows
    object_masks_info = get_object_masks_info2(pred_numpy, class_values)
    symmetry_loss = torch.tensor(0., requires_grad=True).to(device)
    object_masks_info_num = len(object_masks_info)
    for each_object_masks_info in object_masks_info:
        # if each_object_masks_info['stats']
        num_labels =  each_object_masks_info['num_labels']
        labels =  each_object_masks_info['labels']
        stats =  each_object_masks_info['stats']
        centroids =  each_object_masks_info['centroids']
        centroids = torch.tensor(centroids[1:,:].astype(np.float32), requires_grad=True).cuda()
        stats = torch.tensor(stats[1:,:].astype(np.float32), requires_grad=True).cuda()
        # count = 0
        tem_loss = torch.tensor(0., requires_grad=True).to(device)
        index = stats[:, 4] > 100
        num = index.sum()
        if num == 0:
            tem_loss = torch.tensor(0., requires_grad=True).to(device)
            
        else:
            c_hull = torch.zeros((num.item(), 2))
            c_mass = torch.zeros((num.item(), 2))
            c_hull[:,0] = stats[index,0]+(stats[index,2]-1)/2.
            c_hull[:,1] = stats[index,1]+(stats[index,3]-1)/2.
            c_mass[:,0] = centroids[index, 0]
            c_mass[:,1] = centroids[index, 1]
            s_hull = stats[index,2].mul(stats[index,3])
            s_mass = stats[index,4]
            c_loss = torch.mean(torch.sum(torch.square(c_hull-c_mass),dim=1))
            s_loss = torch.mean(torch.square((1/s_hull).mul(s_mass) - 1))
            tem_loss = c_loss + s_loss
        # count = tem_loss.shape[0]
        symmetry_loss = symmetry_loss + tem_loss

    symmetry_loss = symmetry_loss/object_masks_info_num
    loss = loss + 0.1*symmetry_loss
    return loss

def get_object_masks_info2(label_image, class_values):
    '''
    该函数用于实现获取label图像上每个目标对象的mask区域
    input：
        'label_image'：原始的label图像
    return：
        'object_masks_info'：列表，用于存放返回连通区域信息
    '''
    object_masks_info = []  # 初始化
    [c, h, w] = label_image.shape
    for i in range(c):
        masks = np.zeros((h, w), np.uint8)
        for class_value in class_values:
            index = label_image[i,:,:] == class_value
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
            mask_info['id'] = class_value
            object_masks_info.append(mask_info)
    return object_masks_info


def Difference_loss(pred, label):
    torch.diff(pred)