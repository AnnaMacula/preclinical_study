#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:38:15 2022

@author: user
"""

import sys
import numpy as np
import os
# from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.transform import rotate
# from skimage.filters import gaussian
import scipy
# from scipy.ndimage import rotate as ndrotate
# from scipy.ndimage import resize as ndresize
# from scipy.ndimage import crop as ndcrop
# import matplotlib
# from tqdm import tqdm
# import GdDoseReductionModel
# import losses
import io_model
# import psutil
# from sys import getsizeof
import random
from skimage.util import crop
# import itertools
# import SimpleITK as sitk
# from toolz import partition
# from random import sample
# import matplotlib.pyplot as plt
# from skimage.measure import block_reduce
# from skimage.util.shape import view_as_blocks
# import itertools
# from skimage.measure import block_reduce
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count#, IMapIterator, starmapstar
# import time
# import decorators
# import shutil

# import SimpleITK as sitk
# import skimage

local_cwd = os.path.dirname(os.path.realpath("__file__"))
sys.path.append(local_cwd+"/quality metrics")

# import metrics

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PATHS=io_model.configuration()


def DataAugmenter(img, max_rot, flip_h, flip_v, target_size):
    
                   
    if img.shape[0]>target_size[0] and img.shape[1]>target_size[1]:
        dX = img.shape[0]-target_size[0]
        dY = img.shape[1]-target_size[1]
        random_x = random.randrange(dX)
        random_y = random.randrange(dY)
        img = crop(img,((random_x, dX-random_x),(random_y, dY - random_y), (0,0)))
    else:
        img = resize(img, (target_size[0], target_size[1]))# Read an image from folder and resize

    if max_rot is not None:
        img = rotate(img, np.random.uniform(0, max_rot), mode = 'constant', cval = 0)
    if flip_h and np.random.uniform(0,1) > 0.5:
        img = img[::-1,:]
    if flip_v and np.random.uniform(0,1) > 0.5:
        img = img[:,::-1]
        
    return img


class DataAugmentation3D:
    
    def __init__(self, img_shape, target_size=(256,256), max_rot=90, flip_h=True, flip_v=True):
        
        if img_shape[0] != img_shape[1]:
            raise ValueError("Augmentation of rectangular images not implemented.")
        
        dX, dY = img_shape[0] - target_size[0], img_shape[1] - target_size[1]
        
        self.target_size = target_size
        self.dX, self.dY = dX, dY          
        self.hor_flip = flip_h and bool(np.random.choice(2))
        self.vert_flip = flip_v and bool(np.random.choice(2))
        self.rot = np.random.uniform(max_rot)       
        self.dx, self.dy = int(np.random.uniform(0,dX)), int(np.random.uniform(0,dY))
        
        self.larger = img_shape[0]>=target_size[0] and img_shape[1]>=target_size[1]
        
    def rotate(self, imgs):
        
        rotated = scipy.ndimage.rotate(imgs, self.rot, axes=(1,2), reshape=False)

        return rotated
    
    def flip(self, imgs):
        
        flipped = imgs[:,::-1,:,:] if self.hor_flip else imgs  
        flipped = flipped[:,:,::-1,:] if self.vert_flip else flipped
        
        return flipped
    
    def resize(self, imgs):
        
        resized = resize(imgs, output_shape=(imgs.shape[0], *self.target_size, imgs.shape[-1]))
        
        return resized
        
    def crop(self, imgs):
        
        cropped = crop(imgs,((0,0),(self.dx,self.dX-self.dx),(self.dy,self.dY-self.dy),(0,0)))

        return cropped

    def __call__(self, imgs):
        
        imgs = self.crop(imgs) if self.larger else self.resize(imgs) #CHECK DAVIDE      
        augmented = self.rotate(imgs)
        augmented = self.flip(augmented)        
        # augmented = self.crop(augmented) if self.larger else self.resize(augmented)   
                                
        return augmented


def dataGenDoseReduction2D_lowRAM_new_VC(batch_size, image_folder,# range3D=0, 
                                           target_size=(256,256), max_rot=None, flip_h=False, flip_v=False, 
                                        noise_level=0.01, seed=None, max_number_of_studies=5, max_iter_per_studies_block=2, strategy=None, 
                                        strategy_kwargs=None, data_source=None, GPU="0", training_set=None, chunk=["pre", "lowdose","T2","FLAIR", "ADC", "post"]):
    

    patients = io_model.patients_vc(image_folder, GPU=GPU, data_source=data_source, training_set=training_set, chunk=chunk)
    
    
    patients.populate_block(block_length=max_number_of_studies, shuffle=True, strategy=strategy, strategy_kwargs=strategy_kwargs)
    studies_block_images = patients.current_block
    curr_iter_per_studies_block = 0
     
    while True:
        
        if curr_iter_per_studies_block == max_iter_per_studies_block:
            patients.populate_block(block_length=max_number_of_studies, shuffle=True, strategy=strategy, strategy_kwargs=strategy_kwargs)
            studies_block_images = patients.current_block
            curr_iter_per_studies_block = 0
        
        in_channels = np.zeros((batch_size,target_size[0],target_size[1],len(chunk)-1))#2+4*range3D))#solo caso 2D
        out_channel = np.zeros((batch_size,target_size[0],target_size[1],1))
        
        for i in range(batch_size):
            
            study_index = np.random.randint(0, len(studies_block_images))
            slice_index = np.random.randint(0, studies_block_images[study_index].shape[0])
           
            train_img = studies_block_images[study_index][slice_index,:,:,:len(chunk)]#2]

            train_img  = DataAugmenter(train_img, max_rot=max_rot, flip_h=flip_h, flip_v=flip_v, target_size=(256,256))

         
            for i_chunk,label in zip(range(len(chunk)-1),chunk[0:len(chunk)-1]):
                if label != "lowdose":
                    in_channels[i,:,:,i_chunk] = train_img[...,i_chunk]
                else:
                    in_channels[i,:,:,i_chunk] = train_img[...,i_chunk] - train_img[...,0] + np.random.normal(0, noise_level, (in_channels.shape[1], in_channels.shape[2]))
                
         
            out_channel[i,...,0] = train_img[...,-1]  
         
            
        yield in_channels, out_channel  
        curr_iter_per_studies_block += 1
        
        

        
def dataGenDoseReduction2_5D_lowRAM_new(batch_size, image_folder, range3D=0, target_size=(256,256), max_rot=None, flip_h=False, flip_v=False, 
                                        noise_level=0.01, seed=None, max_number_of_studies=5, max_iter_per_studies_block=2, strategy=None, strategy_kwargs=None, data_source=None, GPU="0", training_set=None, chunk=["pre", "highdose"]):
    
    
    patients = io_model.patients_new(image_folder, GPU=GPU, data_source=data_source, training_set=training_set, chunk=chunk)
    
    patients.populate_block(block_length=max_number_of_studies, shuffle=True, strategy=strategy, strategy_kwargs=strategy_kwargs)
    studies_block_images = patients.current_block
    curr_iter_per_studies_block = 0
     
    while True:
        
        if curr_iter_per_studies_block == max_iter_per_studies_block:
            patients.populate_block(block_length=max_number_of_studies, shuffle=True, strategy=strategy, strategy_kwargs=strategy_kwargs)
            studies_block_images = patients.current_block
            curr_iter_per_studies_block = 0
        
        in_channels = np.zeros((batch_size,target_size[0],target_size[1],2+4*range3D))
        out_channel = np.zeros((batch_size,target_size[0],target_size[1],1))
        
        for i in range(batch_size):
            
            study_index = np.random.randint(0, len(studies_block_images))
            slice_index = np.random.randint(range3D, studies_block_images[study_index].shape[0]-range3D)
           
            train_img = studies_block_images[study_index][slice_index-range3D:slice_index+range3D+1,:,:,:2]
            img_hd = np.expand_dims(studies_block_images[study_index][slice_index,:,:,-1], axis=(0,-1))
            
            Augmenter = DataAugmentation3D(img_shape=(train_img.shape[1],train_img.shape[2]), target_size=(256,256), max_rot=90, flip_h=True, flip_v=True)

            train_img, img_hd = np.moveaxis(Augmenter(train_img), 0, -1), Augmenter(img_hd)
         
            
            in_channels[i,...,:2*range3D+1] = train_img[...,0,:]
            in_channels[i,...,2*range3D+1:] = train_img[...,1,:] - train_img[...,0,:] + np.random.normal(0, noise_level, (in_channels.shape[1], in_channels.shape[2], 2*range3D+1))
            
            out_channel[i,...,0] = train_img[...,-1,range3D]  
            out_channel[i,...,0] = img_hd[0,...,0]  

            
        yield in_channels, out_channel  
        curr_iter_per_studies_block += 1
        
def dataGenDoseReduction3D_lowRAM_new(batch_size, image_folder, range3D=1, target_size=(256,256), max_rot=None, flip_h=False, flip_v=False, 
                                        noise_level=0.01, seed=None, max_number_of_studies=5, max_iter_per_studies_block=2, strategy=None, strategy_kwargs=None, data_source=None, GPU="0", training_set=None, chunk=["pre", "highdose"]):
  
    patients = io_model.patients_new(image_folder, GPU=GPU, data_source=data_source, training_set=training_set, chunk=chunk)
    # number_of_studies = patients.n_studies
    
    patients.populate_block(block_length=max_number_of_studies, shuffle=True, strategy=strategy, strategy_kwargs=strategy_kwargs)
    studies_block_images = patients.current_block
    curr_iter_per_studies_block = 0
     
    while True:
        
        if curr_iter_per_studies_block == max_iter_per_studies_block:
            patients.populate_block(block_length=max_number_of_studies, shuffle=True, strategy=strategy, strategy_kwargs=strategy_kwargs)
            studies_block_images = patients.current_block
            curr_iter_per_studies_block = 0
        
        in_channels = np.zeros((batch_size,1+2*range3D,target_size[0],target_size[1],2))
        out_channel = np.zeros((batch_size,1,target_size[0],target_size[1],1))
        
        for i in range(batch_size):
            
            study_index = np.random.randint(0, len(studies_block_images))
            slice_index = np.random.randint(range3D, studies_block_images[study_index].shape[0]-range3D)

            train_img = studies_block_images[study_index][slice_index-range3D:slice_index+range3D+1,:,:,:2]
            img_hd = np.expand_dims(studies_block_images[study_index][slice_index,:,:,-1], axis=(0,-1))
            
            
            Augmenter = DataAugmentation3D(img_shape=(train_img.shape[1],train_img.shape[2]), target_size=(256,256), max_rot=90, flip_h=True, flip_v=True)
            
            train_img, img_hd = Augmenter.augment(train_img), Augmenter.augment(img_hd)
          
            
            in_channels[i,...,0] = train_img[...,0] 
            in_channels[i,...,1] = train_img[...,1] - train_img[...,0] + np.random.normal(0, noise_level, (in_channels.shape[1], in_channels.shape[2], in_channels.shape[3]))
            
             
            out_channel[i,...,0] = img_hd[0,...,0]  
      
            
        yield in_channels, out_channel  
        curr_iter_per_studies_block += 1