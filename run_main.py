import math
import os

import configargparse
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.model import get_model

from utils.dataloader import facade_dataset_collate, facadeDataset
from utils.utils import LossHistory, get_class_info, get_miou_png
from utils.utils_metrics import compute_mIoU, f_score, show_results


def config_parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./configs/MeshFacade.txt',
                        help='config file path')
    parser.add_argument('--expname', type=str, default='building facade', 
                        help='experiment name')
    parser.add_argument('--base_dir', type=str, default='./logs', 
                        help='where to store ckpts and logs')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_period', type=int, default=50)
    parser.add_argument('--fold_name', type=str, default='all')
    parser.add_argument('--input_shape_w', type=int, default=512) 
    parser.add_argument('--input_shape_h', type=int, default=512)
    parser.add_argument('--device', default="cuda:0")

    # train options
    parser.add_argument('--epochs', type=int, default=300, help='epochs of the train')
    parser.add_argument('--model', type=str, default="fcn")
    parser.add_argument('--window', action='store_true', help='weather window')
    parser.add_argument('--MD_Module', action='store_true', help='weather MD_Module')
    parser.add_argument('--MRC_Module', action='store_true', help='weather MRC_Module')
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devive', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--start_epoch', type=int, default=0)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    # set loss
    parser.add_argument('--loss_ignore_0', action='store_false', help='ignore index of 0 when get the loss')  

    # data set
    parser.add_argument('--num_classes', type=int, default=8+1)

    # pred
    parser.add_argument('--pred', action='store_true', help='weather test')
    parser.add_argument('--pred_epoch',default=300, type=int, help='the epoch of pred')
    parser.add_argument('--output_path',default='', help='the path of pred results')

    return parser


def get_data(args):
    with open(os.path.join(args.data_dir, f"ImageSets/train_{args.fold_name}.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.data_dir, f"ImageSets/val_{args.fold_name}.txt"),"r") as f:
        val_lines = f.readlines()
    train_dataset = facadeDataset(train_lines, [args.input_shape_w, args.input_shape_h], args.num_classes, True, args.data_dir)
    val_dataset   = facadeDataset(val_lines, [args.input_shape_w, args.input_shape_h], args.num_classes, False, args.data_dir)
    
    train_dataLoader = DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = facade_dataset_collate)
    val_dataLoader = DataLoader(val_dataset  , shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = facade_dataset_collate)
    return train_dataLoader, val_dataLoader


def save_check_point(model, optimizer, epoch, loss, args):
    if not os.path.exists(os.path.join(args.base_dir, args.expname)):
        os.makedirs(os.path.join(args.base_dir, args.expname))
    path = os.path.join(os.path.join(args.base_dir, args.expname), f'{args.expname}_{args.model}_{args.backbone}_{epoch}.pkl')
    checkpoint = {'model':model.state_dict(), 
                'optimizer':optimizer.state_dict(),
                'epoch':epoch,
                'loss':loss
                }
    torch.save(checkpoint, path)


def load_chech_point_1(model, optimizer, args):
    path = os.path.join(os.path.join(args.base_dir, args.expname), f'{args.expname}_{args.model}_{args.backbone}_{args.start_epoch}.pkl')
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        loss = torch.tensor(0.).ccuda()
        print('############################################ warning ##########################################')
        print('no such weight file', path)
        print('############################################  ##########################################')
    return model, optimizer, loss


def load_chech_point(model, args):
    path = os.path.join(os.path.join(args.base_dir, args.expname), f'{args.expname}_{args.model}_{args.backbone}_{args.pred_epoch}.pkl')
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print('############################################ warning ##########################################')
        print('no such weight file', path)
        print('############################################  ##########################################')
    return model
    

def set_optimizer_lr(optimizer):
    wt = math.pow(1000,-1.0/1000)
    for optim_para in optimizer.param_groups:
            optim_para['lr'] = optim_para['lr']*wt  


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_net(args):
    # load data
    train_dataLoader, val_dataLoader = get_data(args)
    # load model
    model = get_model(args)
    # set gpu device
    device = args.device
    model.to(device)
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    # set loss history
    log_dir = os.path.join(args.base_dir, args.expname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    input_shape = [args.input_shape_w, args.input_shape_h]
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape, val_loss_flag=False)

    # start
    if args.start_epoch != 0:
        print('load the chechpoint')
        model, optimizer, loss = load_chech_point_1(model, optimizer, args)
        model.to(device)
        set_optimizer_lr(optimizer)
        
    # set lossFunction
    if args.loss_ignore_0:
        Cross_loss = nn.CrossEntropyLoss(ignore_index=0)
        print('ignore loss index=0')
    else:
        Cross_loss = nn.CrossEntropyLoss()
        print('ignore loss index=null')
    
    for epoch in range(args.start_epoch+1, args.epochs+1):
        if args.start_epoch>=args.epochs:
            break

        total_loss      = 0
        total_f_score   = 0
        val_loss        = 0
        val_f_score     = 0
        loss_num        = 0

        model.train()
        # set print pbar
        pbar = tqdm(iterable=train_dataLoader, desc=f"{args.expname}_Epoch:{epoch}/{args.epochs}", postfix=dict,mininterval=0.3)
        for iteration, [image, label, seg] in enumerate(train_dataLoader):
            optimizer.zero_grad()
            # set data to device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            seg   = seg.to(device)
            if args.model == 'fcn' or args.model == 'deeplab':
                pred = model(image)['out'] 
            elif args.model == 'Unet' or args.model == 'zero_Unet':
                pred = model(image)
   
            else:
                pass
            # calculate the loss,crossloss
            loss = Cross_loss(pred, label)

            with torch.no_grad():
                    _f_score = f_score(pred, seg)

            loss.backward()
            optimizer.step()
            
            total_loss      += loss.item()
            total_f_score   += _f_score.item()
            loss_num         = iteration + 1
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
        pbar.close()

        loss_history.append_loss(epoch, total_loss/loss_num)
        # save the checkpoint
        if (epoch)%args.save_period == 0:
        # if epoch == args.epochs:
            save_check_point(model, optimizer, epoch, loss, args)
            print(f'done! save check point epoch:{epoch}')
        if (epoch)>=args.epochs -10:
            save_check_point(model, optimizer, epoch, loss, args)
            print(f'done! save check point epoch:{epoch}')
        # updata the optimizer
        set_optimizer_lr(optimizer)
        
    loss_history.writer.close()


def get_pred_result(args):
    print("pred is true")
    building_color, name_classes = get_class_info(args.config)
    image_ids       = open(os.path.join(args.data_dir, f"ImageSets/val_{args.fold_name}.txt"),"r").read().splitlines() 
    gt_dir          = os.path.join(args.data_dir, "Labels/")
    if args.output_path:
        out_path = args.output_path
    else:
        out_path = os.path.join(os.path.join(args.base_dir, args.expname), f'{args.expname}_{args.model}_{args.backbone}_out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pred_dir = os.path.join(out_path, 'pred_png_results')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    # load model
    print("Load model...")
    model = get_model(args)

    # set gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = load_chech_point(model, args)
    model.eval()
    
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(args.data_dir, "Images/"+image_id+".jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(args.data_dir, "Images/"+image_id+".png")
        image = Image.open(image_path)
        image = get_miou_png(image, model, args)

        image.putpalette(building_color)  # give color
        image.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict result done.")

    print("Get miou...")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, args.num_classes, name_classes)  # 执行计算mIoU的函数
    print("Get miou done.")
    show_results(out_path, hist, IoUs, PA_Recall, Precision, name_classes)



def mutil_run():
    parser = config_parse()
    args = parser.parse_args()
    for i in range(1,6):
        args.expname = 'ENPC2014_zero_up2_window_Unet_resnet50_'+f'fold{i}'
        args.fold_name  = f'fold{i}'
        print((args.config))
        if args.pred:
            get_pred_result(args)
        else:
            train_net(args)


if __name__=="__main__":
    parser = config_parse()
    args = parser.parse_args()
    print((args.config))
    if args.pred:
        get_pred_result(args)
    else:
        train_net(args)

    # mutil_run()