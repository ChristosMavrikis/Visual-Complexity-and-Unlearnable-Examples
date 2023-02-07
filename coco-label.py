import cv2

import copy
import os
import collections
import numpy as np
import torch
import util
import random
import mlconfig
import pandas
import imgaug.augmenters as aug
from AddGaussianNoise import AddGaussianNoise
from util import onehot, rand_bbox
from torch.utils.data.dataset import Dataset
from functools import partial
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fast_autoaugment.FastAutoAugment.archive import fa_reduced_cifar10
from fast_autoaugment.FastAutoAugment.augmentations import apply_augment
import imgaug.augmenters as iaa
from torch.utils import data
from pycocotools.coco import COCO


train_transform= transforms.Compose ([
                             transforms.RandomResizedCrop(size=(224,224),scale=(0.5, 1.0)),
                            #transforms.RandomHorizontalFlip(),
                            #transforms.Grayscale(1),
                            np.array])
                            #transforms.ToTensor()])

class MCOCO(data.Dataset):
    def __init__(self, root_dir, ann_file, img_dir, transform ,phase, less_sample=False):

        self.transform = transform
        #initialize image folders/ directories
        self.ann_file = os.path.join(root_dir, ann_file) #annotations directory
        self.img_dir = os.path.join(root_dir, img_dir)   #image directory
        self.coco = COCO(self.ann_file) #
        lst = os.listdir(img_dir) # your directory path
        number_files = len(lst)
        self.img_len = number_files #no. of total images in dir
        #print number_files
        #initiliaze attribute for samples, targets
        self.samples = []
        self.targets = []
        #tmp_ann_ids = self.coco.getAnnIds()
        #tmp_img_ids = self.coco.getImgIds()
        if phase =="train":
            #print(phase)/ceph/csedu-scratch/project/mavrikis/mcoco/unique$
            tmp_img_ids = torch.load('/ceph/csedu-scratch/project/mavrikis/mcoco/ids/low/train/tensor_train.pt')
        else:
            #print(phase)
            tmp_img_ids = torch.load('/ceph/csedu-scratch/project/mavrikis/mcoco/ids/normal/val/tensor_val.pt') #unique high for all unique
        tmp_img_ids = tmp_img_ids.tolist()
        #print(len(tmp_img_ids))
        for img_id in tmp_img_ids:
            img_id = int (img_id)
            tmp_a = self.coco.getAnnIds(imgIds=img_id)#get annotations for img_id
            tmp_cats = []
            tmp_counts = []
            if(len(tmp_a) >0): #if annotations contain more than 0 objects
              ##  print(phase + "here")
                for x in tmp_a:
                    t = self.coco.loadAnns([x])[0] #load annotation x of img_id
                    tmp_label = t['category_id'] #get category_id of annotation x
                    tmp_label = self.label_2_supercategory(tmp_label) #convert category to supercategory - maybe t['supercategory']
                    tmp_cats.append(tmp_label) #append back to list to determine label

                for y in range(len(tmp_cats)): #find most common category
                    tmp_counts.append(tmp_cats[y])
                l = max(tmp_counts)
                self.targets.append(l)
            img_meta = self.coco.loadImgs(img_id)[0]#load image
            img_path = os.path.join(self.img_dir, img_meta['file_name'])
            self.samples.append(img_path)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        img,targets = self.samples[index], self.targets[index]
        #img = Image.open(img).convert('RGB')
        #img = img.resize((224,224)) #resize here otherwise it doesnt resize correctly
        #mg = transforms.Resize(224)(img)
        #if self.transform is not None:
         #   img = self.transform(img)
        return img, targets

    def label_2_supercategory(self,label):
        #person
        if label == 1: return 0
        #vehicle
        elif label >1 and label <=9: return 1
        #outdoor
        elif label >9 and label <=15: return 2
        #animal
        elif label >16 and label <=25:return 3
        #accessory
        elif label >26 and label <=33:return 4
        #sports
        elif label >33 and label <=43:return 5
        #kitchen
        elif label >43 and label <=51:return 6
        #food
        elif label >51 and label <=61:return 7
        #furniture
        elif label >61 and label <=71:return 8
        #electronic
        elif label >71 and label <=77:return 9
        ##appliance
        elif label >77 and label <=83:return 10
        #indoor
        else: return 11
        #elif label >83 and label <=91:return 12
                                                       



train_dataset = MCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
                              ann_file='/ceph/csedu-scratch/project/mavrikis/mcoco/annotations/instances_train2017.json',
                              transform=train_transform,
                              img_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/images-low/train/',
                              #bg_bboxes_file='./box/coco_train_bg_bboxes.log',
                              phase='train')






def rmse(image, compressed):

    """
    Calculates the Root Mean Squared Error

    :param image: original image in ndarray
    :param compressed: compressed image in ndarray
    :return: a float rmse value
    """
    image = image.flatten()
    compressed = compressed.flatten()
    return np.sqrt(((compressed - image)**2).mean())


#data_train = torchvision.datasets.ImageNet(root ='/scratch/data_share_ChDi/Imagenet-100', split='train',transform = train_trans)

j = 0
comp = []
rm = []
for i in range(len(train_dataset.targets)):
    image , l = train_dataset[i]
    image = Image.open(image).convert('RGB')
    image = image.resize((224,224))
    image = np.array(image) 
    path = '/ceph/csedu-scratch/project/mavrikis/2.jpg'
    assert True == cv2.imwrite(path, image)
    image_compressed = cv2.imread(path, 0)
    #qf_.append(estimate_qf(np.array(tmp)))
    comp.append(image.shape[0]*image.shape[1] / os.stat(path).st_size)
    #rm.append( rmse(image ,image_compressed) )

print("compress ratio:" , (sum(comp)/len(comp)))






#rint(train_dataset.targets)
