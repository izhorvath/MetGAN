#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 12:01:29 2020

@author: oschoppe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:23:48 2020

@author: oschoppe
"""

"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset
import os
import pickle
import torch
import numpy as np
import random
import time


class MetganDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=7000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        
        self.folder_cancer =   './data/sample_data/cancer_ch'
        self.folder_labels =  './data/sample_data/labels'
        self.folder_anato =  './data/sample_data/anato_channel'
        
        self.image_paths = os.listdir(self.folder_cancer)
        
        #self.split = opt.dm_fold

        volumes_test = np.load("./data/balanced_test_set_pids.npy")
        samples_for_test = []
        print(len(volumes_test))
        for sample_test in  volumes_test:
            samples_for_test.append('data_patch_'+str(sample_test)+'_X_F15.pickledump')
            samples_for_test.append('data_patch_'+str(sample_test)+'_Y_F15.pickledump')
            samples_for_test.append('data_patch_'+str(sample_test)+'_Z_F15.pickledump')
       
        self.folder_test_cancer =    './data/sample_data/cancer_ch'
        self.folder_test_labels = './data/sample_data/labels'
        self.folder_test_anato =  './data/sample_data/anato_channel'
        
        image_paths_test = os.listdir(self.folder_test_cancer)

       
        self.test_samples =  [image_name for image_name in self.image_paths if image_name in samples_for_test]

        
        self.train_samples = [image_name for image_name in self.image_paths if image_name not in self.test_samples]

        self.phase = opt.phase

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        if(self.phase =='train'):
            sample_cancer = self.train_samples[index]
            folder_cancer = self.folder_cancer 
            folder_anato= self.folder_anato
            folder_label = self.folder_labels
        else:
            sample_cancer = self.test_samples[index]
            folder_cancer =  self.folder_test_cancer 
            folder_anato= self.folder_test_anato
            folder_label = self.folder_test_labels
            

        image_anato = pickle.load(open(os.path.join(folder_anato,sample_cancer)  , "rb" ))['raw']
        image_cancer = pickle.load(open(os.path.join(folder_cancer, sample_cancer)  , "rb" ))['raw']
        label_name = sample_cancer.replace("data", "label")
        label_01 = pickle.load(open(os.path.join(folder_label,label_name)  , "rb" ))
        

        
        #normalize cancer image: 
        image_anato = 2*(image_anato - np.min(image_anato))/(np.max(image_anato)- np.min(image_anato)+1e-10) -1
        image_cancer = 2*(image_cancer - np.min(image_cancer))/(np.max(image_cancer)- np.min(image_cancer)+1e-10)-1
        
         #data is saved normalized between -1 and 1   
        
        
        if(self.phase != 'test'):

            t =  1000*time.time() # current time in milliseconds
            random.seed(int(t))
            rotations = random.randrange(4)
            if(rotations):
                if(rotations ==1):
                    image_cancer = np.rot90(image_cancer)
                    image_anato = np.rot90(image_anato)
                    label_01 = np.rot90(label_01)
                elif (rotations ==2):
                    image_cancer = np.flipud(image_cancer)
                    image_anato = np.flipud(image_anato)
                    label_01 = np.flipud(label_01)
                else:
                    image_cancer = np.fliplr(image_cancer)
                    image_anato = np.fliplr(image_anato)
                    label_01 = np.fliplr(label_01)

	#cropping parts of the image - not the most elegant mode
            
        image_cancer = image_cancer.reshape(1,300,300)[:,:256,:256]
        image_anato = image_anato.reshape(1,300,300)[:,:256,:256]
        label_01 = label_01.reshape(1,300,300)[:,:256,:256]
        

        data_A = torch.Tensor(image_anato.astype(np.float))
        
        data_B = torch.Tensor(image_cancer.astype(np.float))
        
        data_label = torch.Tensor(label_01.astype(np.uint8))
        
        
        return {'A': data_A, 'B': data_B, 'A_paths': sample_cancer, 'B_paths':sample_cancer, 'label': data_label}

    def __len__(self):
        """Return the total number of images."""
        if(self.phase =='train'):
            return len(self.train_samples)
        else:
            return(len(self.test_samples))

