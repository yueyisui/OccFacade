import argparse
import json
import os
import os.path as osp
import warnings
import numpy as np
import PIL.Image
import yaml
from labelme import utils
from PIL import Image

def get_color():

    # 指定包含PNG图像文件的文件夹路径
    folder_path = "/home/yueyisui/YDD/facade/code/OcclusionNet/data/MeshImageFacade/label_color"
    # building_color = [0,0,0, 255,255,0, 255,0,0]
    building_color = [255,255,0, 255,0,0]
    # 遍历文件夹中的PNG图像文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 打开图像
            image = Image.open(file_path)

            # 获取图像的像素数组
            pixel_data = np.array(image)

            # # # 替换像素值
            # # for x in range(pixel_data.shape[0]):
            # #     for y in range(pixel_data.shape[1]):
            # #         if pixel_data[x, y] == 0:
            # #             pixel_data[x, y] = 1
            # #         elif pixel_data[x, y] == 1:
            # #             pixel_data[x, y] = 2

            # # 保存修改后的图像
            image = Image.fromarray(np.uint8(pixel_data))

            image.putpalette(building_color)
            image.save(file_path)
            # print("save to ")
            # print(file_path)

    print("Conversion complete.")

def json2png():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file',default='/home/yueyisui/YDD/facade/code/OcclusionNet/data/MeshImageFacade/json')
    parser.add_argument('--out', default='/home/yueyisui/YDD/facade/code/OcclusionNet/data/MeshImageFacade/png2label')
    args = parser.parse_args()
    json_file = args.json_file
    list = os.listdir(json_file) 
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
            # lbl, lbl_names = utils.shapes_to_label(img.shape, data['shapes'])
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)
            #! choose
            # out_dir = osp.basename(list[i]).replace('.', '_')
            # out_dir = osp.join(args.out, out_dir)
            
            #! else
            out_dir = args.out
            file_name = os.path.splitext(os.path.basename(list[i]))[0]

            if not osp.exists(out_dir):
                os.mkdir(out_dir)
            if not osp.exists(osp.join(out_dir,'label')):
                os.makedirs(osp.join(out_dir,'label'))
            if not osp.exists(osp.join(out_dir,'image')):
                os.makedirs(osp.join(out_dir,'image'))
            if not osp.exists(osp.join(out_dir,'label_viz')):
                os.makedirs(osp.join(out_dir,'label_viz'))
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'image', f'{file_name}_img.png'))
            PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label', f'{file_name}.png'))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz', f'{file_name}_label_viz.png'))
            with open(osp.join(out_dir, f'{file_name}_label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
            print('Saved to: %s' % out_dir)
if __name__ == '__main__':
    # json2png()
    get_color()