import numpy as np
from PIL import Image
from torch.utils import data
import sys
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import pandas as pd

train_data_dir = '/home/zhangyangsong/IMA205/Train/Train/'
train_gt_dir = '/home/zhangyangsong/IMA205/metadataTrain.csv'
random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
    except Exception as e:
        print(img_path)
        raise
    return img

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 45
    angle = random.random() * 2 * max_angle - max_angle
    
    w, h = imgs.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(imgs, rotation_matrix, (h, w))
    imgs = img_rotation
    return imgs

def scale(img, long_size=256):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    random_scale = np.array([0.8, 1.0, 1.2])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs.shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)

    imgs = imgs[i:i + th, j:j + tw, :]

    return imgs

class trainLoader(data.Dataset):
    def __init__(self, is_transform=False, img_size=None):
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)



        self.img_paths = []
        self.gt = []
        
        gts = pd.read_csv(train_gt_dir)

        for idx, img_name in enumerate(gts["ID"]):
            img_path = train_data_dir + str(img_name) + '.bmp'
            img_seg1 = train_data_dir + str(img_name) + '_segCyt.bmp'
            img_seg2 = train_data_dir + str(img_name) + '_segNuc.bmp'
            self.img_paths.append([img_path, img_seg1, img_seg2])
            self.gt.append(gts["ABNORMAL"][idx])    


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        try:
            img_path = self.img_paths[index]
            gt = self.gt[index]

            img = get_img(img_path[0])
            img_seg1 = get_img(img_path[1])
            img_seg2 = get_img(img_path[2])
            img = img*((img_seg1>0) + (img_seg2>0))

            if self.is_transform:
                img = random_scale(img, self.img_size[0])

            if self.is_transform:
                img = random_horizontal_flip(img)
                img = random_rotate(img)
                img = random_crop(img, self.img_size)

            if self.is_transform:
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transforms.ColorJitter(brightness = 5.0 / 255, saturation = 0.1)(img)
            else:
                img = Image.fromarray(img)
                img = img.convert('RGB')

            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            gt = torch.tensor(gt)
            # print(img.shape)
            return img, gt
        except:
            new_index = random.randint(0, len(self) - 1)
            img_path = self.img_paths[new_index]
            gt = self.gt[new_index]
            
            img = get_img(img_path[0])
            img_seg1 = get_img(img_path[1])
            img_seg2 = get_img(img_path[2])
            img = img*((img_seg1>0) + (img_seg2>0))

            if self.is_transform:
                img = random_scale(img, self.img_size[0])

            if self.is_transform:
                img = random_horizontal_flip(img)
                img = random_rotate(img)
                img = random_crop(img, self.img_size)

            if self.is_transform:
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transforms.ColorJitter(brightness = 5.0 / 255, saturation = 0.1)(img)
            else:
                img = Image.fromarray(img)
                img = img.convert('RGB')

            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            gt = torch.tensor(gt)
            # print(img.shape)
            return img, gt

if __name__ == '__main__':
    data_loader = trainLoader(is_transform=True, img_size=256)
    img, gt=data_loader[0]
    print(len(data_loader))
    print(img.shape)
    print(gt)
    img, gt=data_loader[20]
    print(len(data_loader))
    print(img.shape)
    print(gt)
    img=img.transpose(0,1)
    img=img.transpose(1,2)
    cv2.imwrite('img.jpg',np.array(img*255,dtype='uint8'))