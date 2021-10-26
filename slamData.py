#!/usr/bin/python
# -*- encoding: utf-8 -*-


from datetime import time
from genericpath import isfile
from numpy.lib.type_check import imag
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.utils import shuffle

from transform import *

def getPath(root, imgs, labels):
    contents = os.listdir(root)
    for content in contents:
        contPath = os.path.join(root, content)
        if(os.path.isfile(contPath)):
            imgs.append(contPath)
            labels.append(contPath.replace('images', 'labels').replace('.jpg', '.png'))
        else:
            getPath(contPath, imgs, labels)

class parkingSlot(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(parkingSlot, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        self.data = pd.read_csv(rootpth + '/' + self.mode + '.csv', header=None, names=["image","label"])
        self.imgs = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]

        self.len = len(self.imgs)

        self.colors = [(255,255,255), (192,192,0), (113,193,46), (123,64,132), (77,128,255), (255,255,0), (34,134,136), (0,0,0)]
        self.color2id = dict(zip(self.colors, range(len(self.colors))))
        

        # pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            # ColorJitter(
            #     brightness = 0.5,
            #     contrast = 0.5,
            #     saturation = 0.5),
            # HorizontalFlip(),
            # RandomScale(randomscale),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        impth = self.imgs[idx]
        lbpth = self.labels[idx]
        img = Image.open(impth).convert('RGB')
        label = np.array(Image.open(lbpth))
        label = self.convert_labels(label)
        label = Image.fromarray(label)


        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k)-50, np.array(k)+50) == 255] = v
        return mask



if __name__ == "__main__":
    if not os.path.exists('parkingSlot'):
        os.mkdir('parkingSlot')
        images, labels = [], []
        getPath(root='/usr/home/perception/LaneDataset/newver/images', imgs=images, labels=labels)
        all = pd.DataFrame({'image': images, 'label': labels})
        nine_part = int(len(images) * 0.9)
        allshuffled = shuffle(all)
        train_list = allshuffled[:nine_part]
        val_list = allshuffled[nine_part:]
        train_list.to_csv('parkingSlot/train.csv', index=False)
        val_list.to_csv('parkingSlot/val.csv', index=False)
    ds = parkingSlot('/usr/home/ccd/cyr/STDC-Seg-slam/parkingSlot', mode="train")
    platette = np.array(ds.colors).astype(np.uint8)
    for img, lb in ds:
        print(lb.shape)
        label = lb.reshape(lb.shape[1:]).astype(np.uint8)
        label = Image.fromarray(label).convert('P')
        label.putpalette(platette)
        label.save('test.png')
        time.sleep(3)