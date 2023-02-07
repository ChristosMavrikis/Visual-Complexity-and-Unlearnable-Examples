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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Datasets
transform_options = {
     "MCOCO": {
        "train_transform": [
                            transforms.Resize((256,256)),
                            #transforms.RandomResizedCrop(224),
                            transforms.RandomResizedCrop(size=(224,224),scale=(0.5, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((256,256)),
                           transforms.CenterCrop((224,224)),
                           transforms.ToTensor()]},
    "CIFAR10": {
        "train_transform": [
                            transforms.Resize(64),
                            #transforms.GaussianBlur(kernel_size=(31, 31), sigma=(0.1, 0.2)),
                            transforms.RandomCrop(64, padding=4),
                            #transforms.Grayscale(1),
                            #random crop aand padssding 
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(66),transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(20),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "ImageNet": {
        "train_transform": [#transforms.ToTensor(),
                            np.array,
                          iaa.JpegCompression(compression=30).augment_image,
                          transforms.ToPILImage(),
                           #  transforms.Grayscale(3),
                            #transforms.RandomResizedCrop(224),
                            transforms.RandomResizedCrop(224),
                            #transforms.RandomResizedCrop((56,56)),
                            #cluster for standard defender
                         #transforms.RandomEqualize(p=1),
                          #transforms.RandomAdjustSharpness(2, p=1),
                            transforms.RandomHorizontalFlip(),
                            #transforms.Grayscale(3),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                          
                            #transforms.RandomCrop(100),
                           transforms.CenterCrop(224),
                        #   transforms.RandomEqualize(p=1),
                         #                           transforms.RandomAdjustSharpness(2, p=1),
                            #transforms.Grayscale3),
                           transforms.ToTensor()]},
    "TinyImageNet": {
        "train_transform": [transforms.CenterCrop(256),
                            transforms.Resize((32, 32)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((32, 32)),
                           transforms.ToTensor()]},
    'CatDog': {
        "train_transform": [transforms.Resize((32, 32)),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((32, 32)),
                           transforms.ToTensor()]},
    'CelebA': {
        "train_transform": [transforms.CenterCrop((128, 128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.CenterCrop((128, 128)),
                           transforms.ToTensor()]},
    'FaceScrub': {
        "train_transform": [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((128, 128)),
                           transforms.ToTensor()]},
    'WebFace': {
        "train_transform": [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
}
transform_options['PoisonCIFAR10'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR100'] = transform_options['CIFAR100']
transform_options['PoisonCIFAR101'] = transform_options['CIFAR100']
transform_options['PoisonSVHN'] = transform_options['SVHN']
transform_options['ImageNetMini'] = transform_options['ImageNet']
transform_options['PoisonImageNetMini'] = transform_options['ImageNet']
transform_options['MCOCO'] = transform_options['MCOCO']
transform_options['PoisonMCOCO'] = transform_options['MCOCO'] 
transform_options['CelebAMini'] = transform_options['CelebA']


@mlconfig.register
class DatasetGenerator():
    def __init__(self, train_batch_size=128, eval_batch_size=256, num_of_workers=4,
                 train_data_path='../datasets/', train_data_type='CIFAR10', seed=0,
                 test_data_path='../datasets/', test_data_type='CIFAR10', fa=False,
                 no_train_augments=False, poison_rate=1.0, perturb_type='classwise',
                 perturb_tensor_filepath=None, patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None,
                 use_cutout=None, use_cutmix=False, use_mixup=False):

        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        train_transform = transform_options[train_data_type]['train_transform']
        test_transform = transform_options[test_data_type]['test_transform']
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        if no_train_augments:
            train_transform = test_transform

        if fa:
            # FastAutoAugment
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif use_cutout is not None:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

        # Training Datasets
        if train_data_type == 'CIFAR10':
            num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=train_data_path, train=True,
                                             download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR10':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'CIFAR100':
            num_of_classes = 100
            train_dataset = datasets.CIFAR100(root=train_data_path, train=True,
                                              download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR100':
            num_of_classes = 100
            train_dataset = PoisonCIFAR100(root=train_data_path, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise)
        elif train_data_type == 'PoisonCIFAR101':
            num_of_classes = 101
            poison_cifar10 = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise,
                                           poison_classwise_idx=poison_classwise_idx)
            train_dataset = PoisonCIFAR101(train_data_path, split='poison_train',
                                           transform=train_transform, seed=0,
                                           poisn_cifar10_data=poison_cifar10)
        elif train_data_type == 'SVHN':
            num_of_classes = 10
            train_dataset = datasets.SVHN(root=train_data_path, split='train',
                                          download=True, transform=train_transform)
        elif train_data_type == 'PoisonSVHN':
            num_of_classes = 10
            train_dataset = PoisonSVHN(root=train_data_path, split='train', transform=train_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise)
        elif train_data_type == 'TinyImageNet':
            num_of_classes = 1000
            train_dataset = datasets.ImageNet(root=train_data_path, split='train',
                                              transform=train_transform)
        elif train_data_type == 'ImageNetMini':
            #num_of_classes = 100
            num_of_classes=10
            train_dataset = ImageNetMini(root=train_data_path, split='train',
                                         transform=train_transform)
        elif train_data_type == 'PoisonImageNetMini':
            #num_of_classes = 100
            num_of_clasases = 10
            train_dataset = PoisonImageNetMini(root=train_data_path, split='train', seed=seed,
                                               transform=train_transform, poison_rate=poison_rate,
                                               perturb_tensor_filepath=perturb_tensor_filepath)
        elif train_data_type == 'CatDog':
            train_dataset = CatDogDataset(root=train_data_path, split='train',
                                          transform=train_transform)
        elif train_data_type == 'CelebAMini':
            train_dataset = CelebAMini(root=train_data_path, split="all",
                                       target_type="identity", transform=train_transform)
            test_dataset = CelebAMini(root=train_data_path, split="all",
                                      target_type="identity", transform=test_transform)
        elif train_data_type == 'WebFace':
            train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
        elif train_data_type == 'CelebA':
            train_dataset = datasets.CelebA(root=train_data_path, split="all",
                                            target_type="identity", transform=train_transform)
            test_dataset = datasets.CelebA(root=train_data_path, split="all",
                                           target_type="identity", transform=test_transform)
        elif train_data_type =='MCOCO':
            print(train_data_type)
            train_dataset = MCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
                              ann_file='/ceph/csedu-scratch/project/mavrikis/mcoco/annotations/instances_train2017.json',
                              transform=train_transform,
                              img_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/images-high/train/',
                              #bg_bboxes_file='./box/coco_train_bg_bboxes.log',
                              phase='train')
            
            #num_of_classes = 12
            #train_dataset = 
        elif train_data_type == 'PoisonMCOCO':
             train_dataset = PoisonMCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
                              ann_file='annotations/instances_train2017.json',
                              transform=train_transform,
                              img_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/images-high/train/',
                             #bg_bboxes_file='./box/coco_train_bg_bboxes.log',
                              phase='train',seed=0,poison_rate=poison_rate,perturb_tensor_filepath=perturb_tensor_filepath)
            #num_of_classes = 12
            #train_dataset = 
            
        else:
            raise('Training Dataset type %s not implemented' % train_data_type)

        # Test Datset
        if test_data_type == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root=test_data_path, train=False,
                                            download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR10':
            test_dataset = PoisonCIFAR10(root=test_data_path, train=False, transform=test_transform,
                                         poison_rate=poison_rate, perturb_type=perturb_type,
                                         patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                         perturb_tensor_filepath=perturb_tensor_filepath,
                                         add_uniform_noise=add_uniform_noise,
                                         poison_classwise=poison_classwise,
                                         poison_classwise_idx=poison_classwise_idx)

        elif test_data_type == 'CIFAR100':
            test_dataset = datasets.CIFAR100(root=test_data_path, train=False,
                                             download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR100':
            test_dataset = PoisonCIFAR100(root=test_data_path, train=False, transform=test_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise)
        elif test_data_type == 'PoisonCIFAR101':
            test_dataset = PoisonCIFAR101(test_data_path, split='test',
                                          transform=test_transform, seed=0,
                                          poisn_cifar10_data=poison_cifar10)
        elif test_data_type == 'SVHN':
            test_dataset = datasets.SVHN(root=test_data_path, split='test',
                                         download=True, transform=test_transform)
        elif test_data_type == 'PoisonSVHN':
            test_dataset = PoisonSVHN(root=test_data_path, split='test', transform=test_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise)
        elif test_data_type == 'ImageNetMini':
            test_dataset = ImageNetMini(root=test_data_path, split='val',
                                        transform=test_transform)
        elif test_data_type == 'TinyImageNet':
            test_dataset = datasets.ImageNet(root=test_data_path, split='val',
                                             transform=test_transform)
        elif test_data_type == 'PoisonImageNetMini':
            test_dataset = PoisonImageNetMini(root=test_data_path, split='val', seed=0,
                                              transform=test_transform, poison_rate=poison_rate,
                                              perturb_tensor_filepath=perturb_tensor_filepath)
        elif test_data_type =='MCOCO':
            print(test_data_type)
            test_dataset = MCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
                              ann_file='/ceph/csedu-scratch/project/mavrikis/mcoco/annotations/instances_val2017.json',
                              transform=test_transform,
                              img_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/images-high/val/',
                              phase='val')

        elif test_data_type == 'PoisonMCOCO':
            test_dataset = PoisonMCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
                              ann_file='annotations/instances_val2017.json',
                              transform=test_transform,
                              img_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/images-high/val/',
                              phase='val',seed=0,poison_rate=poison_rate,perturb_tensor_filepath=perturb_tensor_filepath)

        elif test_data_type == 'CatDog':
            # Cat Dog only used for transfer exp, no test dataset
            test_dataset = CatDogDataset(root=train_data_path, split='train',
                                         transform=train_transform)
        elif test_data_type == 'CelebAMini' or 'CelebA':
            pass
        elif test_data_type == 'FaceScrub' or test_data_type == 'WebFace':
            pass
        #elif test_data_type =='MCOCO':
         #    num_of_classes = 12
          #   test_dataset = MCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
           #                   ann_file='annotations/instances_val2017.json',
            #                  img_dir='images/val2017',
             #                 phase='val')

        #elif test_data_type == 'PoisonMCOCO':
         #   num_of_classes = 12
          #  test_dataset = PoisonCOCO(root_dir='/ceph/csedu-scratch/project/mavrikis/mcoco/',
           #                   ann_file='annotations/instances_val2017.json',
            #                  img_dir='images/val2017',
             #                 phase='val',seed=0,poison_rate=poison_rate,perturb_tensor_filepath=perturb_tensor_filepath, transform=test_transform)
        else:
            raise('Test Dataset type %s not implemented' % test_data_type)

        if use_cutmix:
            train_dataset = CutMix(dataset=train_dataset, num_class=num_of_classes)
        elif use_mixup:
            train_dataset = MixUp(dataset=train_dataset, num_class=num_of_classes)

        self.datasets = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        return

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

    def _split_validation_set(self, train_portion, train_shuffle=True, train_drop_last=True):
        np.random.seed(self.seed)
        train_subset = copy.deepcopy(self.datasets['train_dataset'])
        valid_subset = copy.deepcopy(self.datasets['train_dataset'])

        if self.train_data_type == 'MCOCO' or self.train_data_type=='PoisonMCOCO':
            print("coco")
            data, targets = self.datasets['train_dataset'].samples, self.datasets['train_dataset'].targets
            print(len(data))
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print((train_D[0]))
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.labels = train_L
            valid_subset.labels = valid_L
            print("Train length: ", len(train_L))
            print("Valid length: ", len(valid_L))

            print("done coco")

        elif self.train_data_type == 'ImageNet' or self.train_data_type == 'ImageNetMini' or self.train_data_type == 'TinyImageNet' or self.train_data_type == 'PoisonImageNetMini':
            print("here i am for portion",train_portion)
            data, targets = list(zip(*self.datasets['train_dataset'].samples))
            datasplit = train_test_split(data, targets, test_size=1-train_portion, 
                                        train_size=train_portion, shuffle=True, stratify=targets) #test_size= 1-train_portion
            train_D, valid_D, train_L, valid_L = datasplit
            print((train_D[0]))
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.samples = list(zip(train_D, train_L))
            valid_subset.samples = list(zip(valid_D, valid_L))
            print("Train length: ", len(train_L))
            print("Valid length: ", len(valid_L))
        elif self.train_data_type == 'SVHN':
            data, targets = self.datasets['train_dataset'].data, self.datasets['train_dataset'].labels
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.labels = train_L
            valid_subset.labels = valid_L
        else:
            datasplit = train_test_split(self.datasets['train_dataset'].data,
                                         self.datasets['train_dataset'].targets,
                                         test_size=1-train_portion, train_size=train_portion,
                                         shuffle=True, stratify=self.datasets['train_dataset'].targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.targets = train_L
            valid_subset.targets = valid_L

        self.datasets['train_subset'] = train_subset
        self.datasets['valid_subset'] = valid_subset
        print(self.datasets)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        data_loaders['train_subset'] = DataLoader(dataset=self.datasets['train_subset'],
                                                  batch_size=self.train_batch_size,
                                                  shuffle=train_shuffle, pin_memory=True,
                                                  drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['valid_subset'] = DataLoader(dataset=self.datasets['valid_subset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)
        return data_loaders


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1: x2, y1: y2, :] = noise
    return mask


class PoisonCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class PoisonCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False):
        super(PoisonCIFAR100, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)

        # Check Shape
        if perturb_type == 'samplewise' and self.perturb_tensor.shape[0] != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 100:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 100))
            self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = self.perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = self.perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            if add_uniform_noise:
                noise = np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] += noise
            self.data[idx] = np.clip(self.data[idx], 0, 255)

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class PoisonCIFAR101(datasets.VisionDataset):
    def __init__(self, root, split='poison_train', transform=None, target_transform=None,
                 poisn_cifar10_data=None, seed=0):
        np.random.seed(seed)
        self.transform = transform
        self.root = root
        if split == 'poison_train':
            self.clean_cifar100 = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
            cifar10 = poisn_cifar10_data
            cifar10_sample_count = 500
        elif split == 'test':
            self.clean_cifar100 = datasets.CIFAR100(root=root, train=False, download=True, transform=None)
            cifar10 = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
            cifar10_sample_count = 100

        self.data, self.targets = self.clean_cifar100.data, self.clean_cifar100.targets
        print(self.clean_cifar100.class_to_idx)
        # Add Ship samples of CIFAR10
        ship_idx = np.where(np.array(cifar10.targets) == 8)[0]
        selected_idx = np.random.choice(ship_idx, cifar10_sample_count, replace=False)
        extra_samples, extra_targets = [], []
        for idx in selected_idx:
            extra_samples.append(cifar10.data[idx])
            extra_targets.append(100)
        self.data = np.concatenate((self.data, np.array(extra_samples)))
        self.targets = self.targets + extra_targets
        self.poison_samples_idx = np.array(range(len(self.clean_cifar100), len(self)))
        self.poison_class = [100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class PoisonSVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False):
        super(PoisonSVHN, self).__init__(root=root, split=split, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        # Check Shape
        if perturb_type == 'samplewise' and self.perturb_tensor.shape[0] != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        self.data = self.data.astype(np.float32)

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            self.poison_samples_idx = []
            for i, label in enumerate(self.labels):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = self.perturb_tensor[idx]
                # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = self.perturb_tensor[self.labels[idx]]
                # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            if add_uniform_noise:
                noise = np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] += noise
            self.data[idx] = np.clip(self.data[idx], 0, 255)

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))



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
            tmp_img_ids = torch.load('/ceph/csedu-scratch/project/mavrikis/mcoco/ids/high/train/tensor_train.pt')
        else:
            #print(phase)
            tmp_img_ids = torch.load('/ceph/csedu-scratch/project/mavrikis/mcoco/ids/high/val/tensor_val.pt') #unique high for all unique
        tmp_img_ids = tmp_img_ids.tolist()
        #print(len(tmp_img_ids))
        for img_id in tmp_img_ids:
            img_id = int (img_id)
            tmp_a = self.coco.getAnnIds(imgIds=img_id)#get annotations for img_id
            tmp_cats = []
            tmp_counts = [] 
            if(len(tmp_a) >=0): #if annotations contain more than 0 objects
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
        img = Image.open(img).convert('RGB')
        #img = img.resize((224,224)) #resize here otherwise it doesnt resize correctly
        #mg = transforms.Resize(224)(img)
        if self.transform is not None:
            img = self.transform(img)
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
        #elif label >83 and label <=91:return 12
        else: return 11


                           
class PoisonMCOCO(MCOCO):
    def __init__(self, root_dir, ann_file, img_dir, transform ,phase, less_sample=False, poison_rate=1.0, seed=0,
             perturb_tensor_filepath=None, **kwargs):

    #def __init__(self,root_dir,  ann_file,img_dir,transform=None,phase, poison_rate=1.0, seed=0,
                 #perturb_tensor_filepath=None):
        super(PoisonMCOCO,self).__init__(root_dir=root_dir, ann_file=ann_file, img_dir=img_dir,transform=transform,phase=phase,**kwargs)
        
        np.random.seed(seed)
        self.poison_rate = poison_rate
        self.perturb_tensor = torch.load(perturb_tensor_filepath)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()

        # Random Select Poison Targets
        targets = list(range(0, len(self)))
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True

        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
        
    def __getitem__(self,index):
        img,targets = self.samples[index], self.targets[index]
        img = Image.open(img).convert('RGB')
        img =  transforms.Resize((256,256))(img)
        #img = np.array(transforms.Resize((224,224))(img)).astype(np.float32)
        #img =  np.array(transforms.Resize((224,224))(img)).astype(np.float32)
        img = np.array(transforms.RandomResizedCrop(size=(224,224),scale = (0.5,1.0))(img)).astype(np.float32)
            #:wimg = img.resize((224,224))
        if self.poison_samples[index]:
            noise = self.perturb_tensor[index]
            #print(noise)
            #print(type(img))
            #print(type(noise))
            img = img + noise
            img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img=Image.fromarray(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    
class ImageNetMini(datasets.ImageNet):
    def __init__(self, root, split='train', **kwargs):
        super(ImageNetMini, self).__init__(root, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        #l = random.sample(range(1,1000),100)
        #l = list()
        #l= [0,510,134,283,346,153,30,555] #similar cifar classes 
        for i, (file, cls_id) in enumerate(self.imgs):
            #if cls_id >99 and cls_id<=199:
            #if cls_id in l:
            #if cls_id <=99:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        print(len(self.samples))
        print(len(self.targets))
        return

class PoisonImageNetMini(ImageNetMini):
    def __init__(self, root, split, poison_rate=1.0, seed=0,
                 perturb_tensor_filepath=None, **kwargs):
        super(PoisonImageNetMini, self).__init__(root=root, split=split, **kwargs)
        np.random.seed(seed)
        self.poison_rate = poison_rate
        self.perturb_tensor = torch.load(perturb_tensor_filepath)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()

        # Random Select Poison Targets
        targets = list(range(0, len(self)))
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True

        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(transforms.RandomResizedCrop(224)(sample)).astype(np.float32)
        #sample = np.array(transforms.RandomResizedCrop((56,56))(sample)).astype(np.float32)
        if self.poison_samples[index]:
           #noise = self.perturb_tensor[target]
            noise = self.perturb_tensor[index]
            sample = sample + noise
            sample = np.clip(sample, 0, 255)
        sample = sample.astype(np.uint8)
        sample = Image.fromarray(sample).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class CatDogDataset(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_file_names = os.listdir(os.path.join(root, split))

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, index):
        filename = self.img_file_names[index]
        label = filename[:3]
        if label == 'cat':
            label = 0
        elif label == 'dog':
            label = 1
        else:
            print(filename)
            raise('Unknown label')

        with open(os.path.join(self.root, self.split, filename), 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class CelebAMini(datasets.CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False, num_of_classes=1000):
        super(CelebAMini, self).__init__(root=root, split=split, target_type=target_type,
                                         transform=transform, target_transform=target_transform,
                                         download=False)

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[datasets.utils.verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)

        mask = slice(None) if split_ is None else (splits[1] == split_)
        identity = identity[mask]
        identity = identity[identity[1] < num_of_classes]
        self.filename = identity.index.values
        self.identity = identity.values
        print(self.identity)

    def __len__(self):
        return len(self.identity)

    def __getitem__(self, index):
        filename = self.filename[index]
        target = self.identity[index][0]
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", filename))
        if self.transform is not None:
            X = self.transform(X)
        return X, target


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class MixUp(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)
