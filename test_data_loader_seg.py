# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import os
test_data_dir = '/home/zhangyangsong/IMA205/Test/Test/'
random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
    except Exception as e:
        print(img_path)
        raise
    return img

def scale(img, long_size=256):
    # h, w = img.shape[0:2]
    # scale = long_size * 1.0 / max(h, w)
    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    img = cv2.resize(img,(256,256))
    return img

class testLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, image_size=2240):
        
        self.img_paths = []
        
        img_names = util.io.ls(test_data_dir, '.bmp')


        img_paths = []
        for idx, img_name in enumerate(img_names):
            name, _ = os.path.splitext(img_name)
            if (name.isdigit()) :
                img_path = test_data_dir + name+'.bmp'
                img_seg1 = test_data_dir + name + '_segCyt.bmp'
                img_seg2 = test_data_dir + name + '_segNuc.bmp'
                
                img_paths.append([img_path, img_seg1, img_seg2])
        
        self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) // part_num
        l = part_id * part_size
        r = (part_id + 1) * part_size
        self.img_paths = self.img_paths[l:r]
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path[0])
        img_seg1 = get_img(img_path[1])
        img_seg2 = get_img(img_path[2])
        img = img*((img_seg1>0) + (img_seg2>0))

        scaled_img = scale(img, self.image_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        return scaled_img

if __name__ == '__main__':
    data_loader = testLoader(image_size=256)
    print(len(data_loader))
    img=data_loader[0]
    
    print(img.shape)
    # img, gt=data_loader[20]
    # print(len(data_loader))
    # print(img.shape)
    # img=img.transpose(0,1)
    # img=img.transpose(1,2)
    # cv2.imwrite('img.jpg',np.array(img*255,dtype='uint8'))