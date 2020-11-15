import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import math
import random
import time


class Camvid_Train_DataSet():

    def __init__(self):
        self.images = []
        self.labels = []
        self.images_path = "../image/images"
        self.labels_path = "../image/new_labels"
        # 该数据集共701张图片，我们用前500训练
        self.used_number = 500
        self.size = 0
        self.index = 0
        self.get_image_path()

    def get_image_path(self):
        names = os.listdir(self.images_path)[:self.used_number]
        for name in names:
            self.images.append(self.images_path+"/"+name)
            self.labels.append(self.labels_path+"/"+name)
        self.size = len(names)

    def get(self):
        image = cv2.imread(self.images[self.index])
        label = cv2.imread(self.labels[self.index])[:, :, 0]
        self.index+=1
        self.index%=self.size
        return image, label

class Camvid_Test_DataSet():

    def __init__(self):
        self.images = []
        self.labels = []
        self.images_path = "../image/images"
        self.labels_path = "../image/new_labels"
        # 该数据集共701张图片，我们用前500训练, 后201张用于测试
        self.used_number = 500
        self.size = 0
        self.index = 0
        self.get_image_path()

    def get_image_path(self):
        names = os.listdir(self.images_path)[self.used_number:]
        for name in names:
            self.images.append(self.images_path+"/"+name)
            self.labels.append(self.labels_path+"/"+name)
        self.size = len(names)

    def get(self):
        image = cv2.imread(self.images[self.index])
        label = cv2.imread(self.labels[self.index])[:, :, 0]
        self.index+=1
        self.index%=self.size
        return image, label

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

def DataAugmentation(image, label, crop_size=320):

    # crop
    h, w, c = image.shape
    # 这个数据集的道路都在下边，所以尽可能往下
    x1 = np.random.randint(0, w - crop_size)
    y1 = np.random.randint(200, h - crop_size)
    # print(x1, y1)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    image = crop_hwc(image, [x1, y1, x2, y2], crop_size)
    label = crop_hwc(label, [x1, y1, x2, y2], crop_size)



    # flip
    flip_style = np.random.randint(-1, 1)
    # print(flip_style)
    image = cv2.flip(image, flip_style)
    label = cv2.flip(label, flip_style)

    # cv2.imshow("image", image)
    # cv2.imshow("label", label*255)
    # cv2.waitKey()
    return image, label

class Trian_DataSet(Dataset):
    def __init__(self):
        super(Trian_DataSet, self).__init__()
        print("dataset init start")
        self.data_set = Camvid_Train_DataSet()
        self.size = self.data_set.size
        self.set_numpy_seed()
        print("dataset init done")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image, label = self.data_set.get()
        image, label = DataAugmentation(image, label)
        label = np.expand_dims(label, -1)
        # print(image.shape)
        # print(label.shape)

        image, label = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [image, label])
        # print(image.shape)
        # print(label.shape)
        return image, label

    def set_numpy_seed(self):
        np.random.seed(int(time.time()))

class Test_DataSet(Dataset):
    def __init__(self):
        super(Test_DataSet, self).__init__()
        print("dataset init start")
        self.data_set = Camvid_Test_DataSet()
        self.size = self.data_set.size
        self.set_numpy_seed()
        print("dataset init done")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image, label = self.data_set.get()
        image, label = DataAugmentation(image, label)
        label = np.expand_dims(label, -1)
        # print(image.shape)
        # print(label.shape)

        image, label = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [image, label])
        # print(image.shape)
        # print(label.shape)
        return image, label

    def set_numpy_seed(self):
        np.random.seed(int(time.time()))

if __name__ == '__main__':
    a = Test_DataSet()
    image, label = a.__getitem__(1)
    # label*=255
    # cv2.imshow("image", image)
    # cv2.imshow("label", label)
    # cv2.waitKey()