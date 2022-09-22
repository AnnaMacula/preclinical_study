#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:11:35 2022

@author: user
"""
import os 
# import pandas as pd
# import sys
# from difflib import SequenceMatcher
# import h5py
# from skimage.io import imread
import SimpleITK as sitk
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from toolz import partition
# import models
# from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16, VGG19

import itertools
# from tensorflow.keras.models import Model

from inspect import getmembers, isfunction
from tqdm import tqdm
import losses

import json
import SIM_strategies
from pathlib import Path

import locale

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def path_trimming(path, trim_to):
    
    if trim_to in path: 
        path = Path(path)
    else:
        raise ValueError(f"<{trim_to}> not in <{path}>")
    while path.name != trim_to:
        path = path.parent
        # print(path)
    return path


def configuration(local_cwd=os.path.dirname(os.path.realpath("__file__")), config_folder="Preclinical_Study"):
        
    relative_path = path_trimming(local_cwd, config_folder)
    
    with open(os.path.join(relative_path,'local_configuration.txt')) as json_file:
        PATHS = json.load(json_file)
    
    return PATHS["PATHS"]

PATHS=configuration()


default_path=PATHS["path_trained_models"]


def modelcut_new(network_label, input_shape = (256, 256, 3)):
    
    model, layer = network_label.split("-")    
    new_input = tf.keras.Input(shape=input_shape)
   
    if "VGG16" in model:        
        j = int(layer)*4-2
        modelall = VGG16(include_top=False, weights='imagenet', input_tensor=new_input)
        
    elif "VGG19" in model: 
        j = int(layer)*5-4               
        modelall = VGG19(include_top=False, weights='imagenet', input_tensor=new_input)

    modelcut = tf.keras.Sequential(modelall.layers[:j+1])   
    
    name = "Perceptual_" + network_label if "Perceptual_" not in network_label else network_label
    
    return modelcut, name#, input_shape[-1]



class normalization_class:
    
    def __init__(self):
    
        preclinic2020 = {
                "MAE" : 79.08,
            "FFT_MAE" : 0.364,        
            "Perceptual_VGG16-3" : 0.411,
            "Perceptual_VGG16-4" : 9.507,
            "Perceptual_VGG16-5" : 382.574,
            "Perceptual_VGG19-3" : 0.337,
            "Perceptual_VGG19-4" : 3.047,
            "Perceptual_VGG19-5" : 594.984,
            "GAN_VGG16" : 0.073,
            "GAN_pix2pix" : 1.0}
        
        preclinic2021_2D = {
            "MAE" : 100.21,
        "FFT_MAE" : 0.433,
        "Perceptual_VGG19-4" : 15.07}
        
        preclinic2021_3D = {
            "MAE" : 173.273,
        "FFT_MAE" : 0.803,
        "Perceptual_VGG19-4" : 19.207}
        
        clinic_cns = {
        "MAE" : 48.328,
        "FFT_MAE" : 0.371,
        "Perceptual_VGG19-4" : 1.821,   
        "GAN_VGG16" : 0.672}
        
        clinic_liver = {
            "MAE" : 42.534,
        "FFT_MAE" : 0.527,
        "Perceptual_VGG19-4" : 5.905}
        
        
        clinic_ctcns = {
            "MAE" : 177.338,
        "FFT_MAE" : 0.841,
        "Perceptual_VGG19-4" : 7.728}
        
        clinic_vc = {
            "MAE" : 13.884, #(21.02, normalizzazione con 289 pazienti, 6 esclusi per Skull Stripping errato)
        "FFT_MAE" : 0.131,
        "Perceptual_VGG19-4" : 0.417}
        
        clinic_hbp = {
          "MAE" : 26.46,
      "FFT_MAE" : 0.436,
      "Perceptual_VGG19-4" : 1.038}
     
        self.dic = {"preclinic2020" : preclinic2020,
                    "preclinic2021_2D" : preclinic2021_2D,
                    "preclinic2021_3D" : preclinic2021_3D,
               "clinic_cns" : clinic_cns,
               "clinic_liver" : clinic_liver,
               "clinic_ctcns" : clinic_ctcns,
               "clinic_vc" : clinic_vc,
               "clinic_hbp" : clinic_hbp}
  

        
    
    def coefficients(self, data_source, normalize=True):
        
        return self.dic[data_source]
    
    def save_txt(self, path):
        
        with open(f"{path}/normalization_coeffs.txt","w") as f:
            json.dump(self.dic, f)
        
        
        
        

def parse_loss_new(*args, normalize=True, data_source, **kwargs):
    
    onetoone = {parameter : [arg, value] for [parameter,value], arg in zip(kwargs.items(), args)}
    
    funcs1 = {parameter : [func, func.__name__, value] for parameter, [func, value] in onetoone.items() if callable(func) and value is not None}  
    
    
    found = {name : func for (name, func), arg in itertools.product(getmembers(losses, isfunction), 
                                             args) if isinstance(arg, str) and name in arg}        
    
    funcs2 = {parameter : [func, func.__name__, value] for (arg, func), (parameter, [name, value]) in itertools.product(found.items(), 
                                             onetoone.items()) if name == arg and value is not None}
  
    active_losses = {**funcs1, **funcs2}
    
    normalization = normalization_class()
    
    norm = {value[1] : normalization.coefficients(data_source)[value[1]] for value in active_losses.values()}

    if not normalize:
        norm = {value[1] : 1.0 for value in active_losses.values()}  
    
    for parameter in sorted(active_losses.keys()):
        func, name, value = active_losses[parameter]    
        print(f"{name} component active with coefficient {parameter}={kwargs[parameter]}, normalized using {norm[name]}.\n")

    components = {name : [func, value*norm[name]] for key, [func, name, value] in active_losses.items()}
    
    loss_functions = losses.standard_loss_composition(components)    
    
    print("Components order: ", *loss_functions.keys())
    
    additional_info = {}
    additional_info["coupling coeffs"] = [arg or 0.0 for arg in kwargs.values()] #if_null_then_default(*kwargs.values(), default=0.0)    
    additional_info["normalization coefficients"] = list(norm.values()) 
 
    return loss_functions, additional_info


class patients_vc:
    
    def __init__(self, image_folder, GPU="0", data_source=None, training_set=None, chunk=["pre","FLAIR","ADC","T2", "post"]):
        
        #GPU="1" if "clinic" in data_source.lower() else GPU #SCS commentaat
        if chunk[0] != "pre":
            raise ValueError("ERROR!!!: first image in list has to be a pre contrast")
        #pensso che non vada cancellata la parte commentata (inckusa parte commentata)
        if image_folder.lower()=="test":
            image_folder = PATHS[f"path_test_{data_source.lower()}"]
        elif image_folder.lower()=="train":
            image_folder = PATHS[f"path_train_{data_source.lower()}"] + f"/data_for_training_{training_set.lower()}_{GPU}/train"
        elif image_folder.lower()=="val" or image_folder.lower()=="validation" :
            image_folder = PATHS[f"path_train_{data_source.lower()}"] + f"/data_for_training_{training_set.lower()}_{GPU}/val"        
        #print(image_folder)
        files = os.listdir(image_folder)
        #print(files)
        sorted_files = [[imageFile for imageFile in sorted(files) if label in imageFile] for label in chunk]
        
        partitioned = [study for study in zip(*sorted_files)]
        
        self.image_folder = image_folder
        self.files = partitioned
        self.n_studies = len(partitioned)                 
        self.training_set = training_set or "SIM"
        self.chunk = chunk
        self.current_block = None
        self.names = None
        
        print(f"{self.n_studies} studies found.")
   
        
       #qua dobbiamo fare una lista di array
    def populate_block(self, images_files_list=None, block_length=None, shuffle=True, quiet=False, strategy=None, strategy_kwargs=None):
        
        block_length = block_length or self.n_studies
        
        strategy, strategy_kwargs = strategy or "standard", strategy_kwargs or []
        strategy = eval(f"SIM_strategies.{strategy}(*strategy_kwargs)")
        
        partitioned = self.files if images_files_list is None else list(partition(len(self.chunk), sorted(images_files_list)))
        
        if shuffle:
            np.random.shuffle(partitioned)
        
        images_as_list = []
        names = []
        for study in tqdm(partitioned[:block_length], desc="populating block"):
            dic = {}    
            for imageFile in study:
                ##################
                for label in self.chunk:
                    if label in imageFile:
                        image = sitk.ReadImage(os.path.join(self.image_folder, imageFile))            
                        dic[label] = sitk.GetArrayFromImage(image)
                        if label == "pre":
                            name = imageFile.replace("_T1_pre.nrrd", "")
            names.append(name)
                
     
            images_list_to_tuple = []
            for label in self.chunk:
                images_list_to_tuple.append(dic[label])
            images_as_tuple = tuple(images_list_to_tuple)
                        
            images_as_list.append(np.stack(images_as_tuple, axis=-1))
             
        if not quiet:
            print("New patients block ready.")
        
        self.current_block = images_as_list
        self.names = names



class patients_new:
    
    def __init__(self, image_folder, GPU="0", data_source=None, training_set=None, chunk=["pre", "highdose"], samba=False):
        
        #GPU="1" if "clinic" in data_source.lower() else GPU #SCS commentaat
        
        #pensso che non vada cancellata la parte commentata (inckusa parte commentata)
        if image_folder.lower()=="test":
            image_folder = PATHS[f"path_test_{data_source.lower()}"]
        elif image_folder.lower()=="train":
            image_folder = PATHS[f"path_train_{data_source.lower()}"] + f"/data_for_training_{training_set.lower()}_{GPU}/train"
        elif image_folder.lower()=="val" or image_folder.lower()=="validation" :
            image_folder = PATHS[f"path_train_{data_source.lower()}"] + f"/data_for_training_{training_set.lower()}_{GPU}/val"        
        #print(image_folder)
        files = os.listdir(image_folder)
        #print(files)
        sorted_files = [[imageFile for imageFile in sorted(files) if label in imageFile] for label in chunk]
        
        partitioned = [study for study in zip(*sorted_files)]
        
        self.image_folder = image_folder
        self.files = partitioned
        self.n_studies = len(partitioned)                 
        self.training_set = training_set or "SIM"
        self.chunk = chunk
        self.current_block = None
        self.names = None
        
        print(f"{self.n_studies} studies found.")
        
    def populate_block(self, images_files_list=None, block_length=None, shuffle=True, quiet=False, strategy=None, strategy_kwargs=None):
        
        block_length = block_length or self.n_studies
        
        strategy, strategy_kwargs = strategy or "standard", strategy_kwargs or []
        strategy = eval(f"SIM_strategies.{strategy}(*strategy_kwargs)")
        
        partitioned = self.files if images_files_list is None else list(partition(len(self.chunk), sorted(images_files_list)))
        
        if shuffle:
            np.random.shuffle(partitioned)
        
        images_as_list = []
        names = []
        for study in tqdm(partitioned[:block_length], desc="populating block"):
                
            for imageFile in study:
                if "pre" in imageFile:
                    image = sitk.ReadImage(os.path.join(self.image_folder, imageFile))            
                    pre = sitk.GetArrayFromImage(image)
                    name = imageFile.replace("_pre.nrrd", "")
                elif "lowdose" in imageFile:
                    image = sitk.ReadImage(os.path.join(self.image_folder, imageFile))
                    ld = sitk.GetArrayFromImage(image) 
                elif "highdose" in imageFile or "post" in imageFile:
                    image = sitk.ReadImage(os.path.join(self.image_folder, imageFile))
                    hd = sitk.GetArrayFromImage(image)
                          
            # dic = {}
            # for imageFile, label in zip(study, self.chunk):
            #     image = sitk.ReadImage(os.path.join(self.image_folder, imageFile))      
            #     dic[label] = sitk.GetArrayFromImage(image)
            
            
            # pre, hd = dic["pre"], dic["highdose"]
            
            ld = strategy(pre, hd) if "lowdose" not in self.chunk else ld #dic["lowdose"]             
            #SCS nota: da come capisco fai cmq una seconda normalizzazione, capisco che dovrebbe dividire per 1, ma da controllare
            
            pre, ld, hd = SIM_strategies.normalization(pre, ld, hd) 
            
            names.append(name)
            images_as_list.append(np.stack((pre, ld, hd), axis=-1))
             
        if not quiet:
            print("New patients block ready.")
        
        self.current_block = images_as_list
        self.names = names
   
    def as_3D(self, patient, range3D=1):
        
        if self.current_block is None:
            print("Patients block not yet populated. Populate block first.")
            raise RuntimeError
        
        study = self.current_block[patient]
        
        vol_size = study.shape
        slc_range = range(range3D,vol_size[0]-range3D)# if slc is None else range(slc,slc+1)
        if patient+1>self.n_studies:
            print(f"patient {patient} does not exist. Patients: {self.n_patients}.")                
            raise OSError
            
        full_img = np.zeros((len(slc_range),1+2*range3D,vol_size[1],vol_size[2],3))

        for k in slc_range:              
            index = k-range3D# if slc is None else 0
            full_img[index,...,0] = study[k-range3D:k+range3D+1,...,0]      
            full_img[index,...,1] = study[k-range3D:k+range3D+1,...,1] - study[k-range3D:k+range3D+1,...,0]      
            full_img[index,...,-1] = study[k-range3D:k+range3D+1,...,-1]

        return full_img


    def as_2_5D(self, patient, range3D=1):
        
        if self.current_block is None:
            print("Patients block not yet populated. Populate block first.")
            raise RuntimeError
       
        study = self.current_block[patient]
        
        vol_size = study.shape        
        slc_range = range(range3D,vol_size[0]-range3D)

        if patient+1>self.n_patients:
            print(f"patient {patient} does not exist. Patients: {self.n_patients}.")                
            raise OSError
            
        in_channels = np.zeros((len(slc_range),self.vol_size[1],self.vol_size[2],2+4*range3D))
        
        for slc in slc_range:
            in_channels[slc-range3D,...,:2*range3D+1] = np.moveaxis(study[slc-range3D:slc+range3D+1,...,0], 0, -1)  
            in_channels[slc-range3D,...,2*range3D+1:] = np.moveaxis(study[slc-range3D:slc+range3D+1,...,-2], 0, -1) - np.moveaxis(study[slc-range3D:slc+range3D+1,...,0], 0, -1)
            
        out_channel = study[slc_range[0]:slc_range[-1]+1,...,-1] 
        full_img = np.concatenate((in_channels, out_channel), axis=-1)

        return full_img  