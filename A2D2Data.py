#!/usr/bin/python
# -*- encoding: utf-8 -*-
from datetime import time
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import pandas as pd
import time

from transform import *

def getPath(root, imgs, labels):
    contents = os.listdir(root)
    for content in contents:
        contPath = os.path.join(root, content)
        if(os.path.isfile(contPath)):
            imgs.append(contPath)
            labels.append(contPath.replace('image', 'label'))
        else:
            getPath(contPath, imgs, labels)

def hex2rgb(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return (r, g, b)

def getColor2id(file = 'class_list.json'):
    with open('./class_list.json', 'r') as f:
        clslist = json.load(f)
    hexcolors = list(clslist.keys())
    colors = []
    for hex in hexcolors:
        colors.append(hex2rgb(hex))
    color2id = dict(zip(colors, list(clslist.values())))
    return colors, color2id

class AudiDataset(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(AudiDataset, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        # self.data = pd.read_csv(rootpth + '/' + self.mode + '.csv', header=None, names=["image","label"])
        # self.imgs = self.data["image"].values[1:]
        # self.labels = self.data["label"].values[1:]
        self.data = [line.strip().split() for line in open("/nfs/data/A2D2/split" + '/' + self.mode + '.lst')]
        self.imgs = [i for i, _ in self.data]
        self.labels = [l[::-1].replace('label'[::-1],'mask'[::-1],2)[::-1] for _, l in self.data]

        self.len = len(self.imgs)
        self.colors, self.color2id = getColor2id()
        

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
        lbpth = self.labels[idx].replace('label', 'mask')
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        # label = np.array(Image.open(lbpth))
        # label = self.convert_labels(label)
        # label = Image.fromarray(label)


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
        mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for k, v in self.color2id.items():
            mask[(label==k).all(axis=2)] = v
        return mask

if __name__ == "__main__":
    if not os.path.exists('A2D2'):
        os.mkdir('A2D2')
        # train dataset path
        trainimages, trainlabels = [], []
        getPath(root='/nfs/data/A2D2/split/training/image', imgs=trainimages, labels=trainlabels)
        all = pd.DataFrame({'image': trainimages, 'label': trainlabels})
        all.to_csv('A2D2/train.csv', index=False)
        # test dataset path
        valimages, vallabels = [], []
        getPath(root='/nfs/data/A2D2/split/testing/image', imgs=valimages, labels=vallabels)
        all = pd.DataFrame({'image': valimages, 'label':vallabels})
        all.to_csv('A2D2/val.csv', index=False)
        

    ds = AudiDataset('./A2D2', mode="val")
    platette = np.array(ds.colors).astype(np.uint8)
    for img, lb in ds:
        print(lb.shape)
        label = lb.reshape(lb.shape[1:]).astype(np.uint8)
        label = Image.fromarray(label).convert('P')
        label.putpalette(platette)
        label.save('test.png')
        exit()
        # time.sleep(3)