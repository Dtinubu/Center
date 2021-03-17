import os
import random
import re
import tarfile
import sys
import itertools
import random
from math import ceil, floor


from torch.utils import data
import numpy as np

from utils import image_loader, download


def create_datasets(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)

    dataroot_files = os.listdir(dataroot)
    data_tarball_file = []
    data_dir_name = []

    if data_dir_name not in dataroot_files:
        if data_tarball_file not in dataroot_files:
            tarball = download(dataroot, DATASET_TARBALL)
        with tarfile.open(tarball, 'r') as t:
            t.extractall(dataroot)




def create_datasetsAF(af_pics, dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot,'RFW-deepfunneled.tar')


    images_root = os.path.join(dataroot, 'African')
    names_af = os.listdir(images_root)
    if len(names_af) == 0:
        raise RuntimeError('Empty dataset')

    af_t_training_set =[]
    af_t_validation_set =[]
    
  
    for klass, name in enumerate(names_af):  
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)
        
        images_of_person = os.listdir(os.path.join(images_root, name))     
        total = len(images_of_person)
        
        def fold(folds, data):
            tot_length = len(data)
            split_length = tot_length // folds
    
    for i in range(folds):
        train_dataset = DatasetSplit(data, (i + 1) * split_length, tot_length - split_length)
        val_dataset = DatasetSplit(data, i * split_length, split_length)
        yield (train_dataset, val_dataset)
             
    return af_training_set, af_validation_set, len(names_af)
  
        
def create_datasetsR(race, Number_of_pics, dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)


    images_root = os.path.join(dataroot, race)
    names = os.listdir(images_root)
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    train_dataset = []
    val_dataset = []
    
        for klass, name in enumerate(names_w):
             for i in range (Number_of_pics):
                def add_class(image):
                    image_path = os.path.join(images_root, name, image)
                    return (image_path, klass, name)
            images_of_person = os.listdir(os.path.join(images_root, name))
            total = len(images_of_person)

            whole_set += map(
                    add_class,
                    images_of_person)
            def fold(folds, data):
                tot_length = len(data)
                split_length = tot_length // folds
                for i in range(folds):
                      train_dataset = whole_set(data, (i + 1) * split_length, tot_length - split_length)
                      val_dataset = whole_set(data, i * split_length, split_length)
            yield (train_dataset, val_dataset)
    return val_dataset, train_dataset, len(names)
          
    
def create_datasets(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)

    training_set = []
    validation_set = []
    count=0
    for count in range(0, 1):
        for klass, name in enumerate(names):
            count+=1
            def add_class(image):
                image_path = os.path.join(images_root, name, image)
                return (image_path, klass, name)
            images_of_person = os.listdir(os.path.join(images_root, name))
            total = len(images_of_person)

            training_set += map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
            validation_set += map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])
            w_t_training_set += map(
                    add_class,
                    images_of_person[:int(total * train_val_split)])
            w_t_validation_set += map(
                    add_class,
                    images_of_person[int(total * train_val_split):])
    random.shuffle(w_t_validation_set)
    random.shuffle(w_t_training_set)
    w_validation_set = w_t_validation_set[0:int(Number_of_pics*.10)]
    w_training_set = w_t_training_set[0:int(Number_of_pics*.90)]          
    return w_training_set, w_validation_set, len(names_w)

    return training_set, validation_set, len(names)



class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])
    
class PairedDataset(data.Dataset):

    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader

        self.image_names_a = []
        self.image_names_b = []
        self.matches = []
       
        self._prepare_dataset()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        return (self.transform(self.loader(self.image_names_a[index])),
                self.transform(self.loader(self.image_names_b[index])),
                self.matches[index])

    def _prepare_dataset(self):
        raise NotImplementedError
        
class LFWPairedDataset(PairedDataset):

    def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)

        for pair in pairs:
            if len(pair) == 3:
                match = True
                name1, name2, index1, index2 = \
                    pair[0], pair[0], int(pair[1]), int(pair[2])

            else:
                match = False
                name1, name2, index1, index2 = \
                    pair[0], pair[2], int(pair[1]), int(pair[3])

            self.image_names_a.append(os.path.join(
                    self.dataroot,
                    name1, "{}_{:04d}.jpg".format(name1, index1)))

            self.image_names_b.append(os.path.join(
                    self.dataroot,
                    name2, "{}_{:04d}.jpg".format(name2, index2)))
            self.matches.append(match)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines():
                pair = line.strip().split()
                pairs.append(pair)
        return pairs
