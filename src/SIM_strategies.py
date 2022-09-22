#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:32:08 2021

@author: user
"""
from scipy.ndimage import gaussian_filter
import numpy as np
import tensorflow as tf
from keras.models import model_from_json


def k_func(enhmap, slope=5.0, intercept=0.6):   
    
    karray = intercept - slope*enhmap
    karray[karray>5.0]=5.0
    karray[karray<2.0]=5.0

    return karray

def normalization(pre, ld, hd):
    
    normFactor = np.max(ld)
    ImagePre = pre/normFactor
    ImageLowDose = ld/normFactor
    ImageHighDose = hd/normFactor

    return ImagePre, ImageLowDose, ImageHighDose#, ImageEnhancement#, normFactor



def threshold_mask(threshold, image):    
    
    mask = np.ones((24,256,256), dtype="bool")    
    mask = image>threshold
    
    return mask


def k_modulation(slope=5.0, intercept=0.6):

    def strategy(pre, hd):
        
        highdose_blurred = gaussian_filter(hd, sigma=1)
        mask = threshold_mask(4, highdose_blurred)
        enhmap = (hd-pre)/pre 
        enhmap[~mask] = 0.0
        karray = k_func(enhmap, float(slope), float(intercept))
        lowdose = pre + (hd-pre)/karray   
    
        return lowdose

    return strategy


def standard(k=5):

    def strategy(pre, hd):
        
        lowdose = pre + (hd-pre)/float(k)
      
        return lowdose
    
    return strategy


json_path="/home/user/Scrivania/Preclinical Study/Local_Repository/GDDoseReduction/src/DnCNN/model.json"
model_path="/home/user/Scrivania/Preclinical Study/Local_Repository/GDDoseReduction/src/DnCNN/model.h5"

def json_load(json_path=json_path, model_path=model_path):
    
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_path)
        
    return model

def dnCNN(k=5):
    
    def strategy(pre_dn, hd_dn):
        
        pre_dn = np.nan_to_num(pre_dn, copy=False)
        hd_dn = np.nan_to_num(hd_dn, copy=False)
    
        lowdose = pre_dn + (hd_dn - pre_dn)/float(k)
       
        return lowdose
    
    return strategy

def network_preprocessing(pre, hd, model_path, func="standard", func_args=None):#, json_load=False, **load_kwargs):
     
    strategy = eval(f"{func}")
    strategy_args = list(map(float, func_args))
    
    # if json_load:
    #     model = json_load(**load_kwargs)
    # else:
    model = tf.keras.models.load_model(model_path, compile=False)
    
    pre_processed = model.predict(pre, batch_size=24)[:,:,:,0]
    hd_processed = model.predict(hd, batch_size=24)[:,:,:,0]

    ImagePre, ImageLowDose, ImageHighDose = strategy(pre_processed, hd_processed, *strategy_args)
    
    normFactor = np.max(ImageLowDose)
    
    ImagePre = pre/normFactor
    ImageLowDose = ImageLowDose/normFactor
    ImageEnhancement = ImageLowDose-ImagePre
    
    return ImageEnhancement, ImageLowDose, normFactor


def network_prediction(model_path, k=5):
    
    def strategy(pre, hd):
        
        model = tf.keras.models.load_model(model_path, compile=False)
    
        hd_pred = model.predict(hd, batch_size=24)[:,:,:,0]   
        pre_pred = model.predict(pre, batch_size=24)[:,:,:,0]
           
        ld = standard(float(k))(pre_pred,hd_pred)[2]
          
        return ld
    
    return strategy

def PN2V(k=5):
    
    def strategy(pre_PN2V, hd_PN2V):
    
        pre = pre_PN2V[:,:,:,0]
        hd = hd_PN2V[:,:,:,0]
        
        
        pre = np.nan_to_num(pre, copy=False)
        hd = np.nan_to_num(hd, copy=False)
    
        lowdose = standard(float(k))(pre,hd)[2]
   
        return lowdose
    
    return strategy

def PN2V_alt(pre_PN2V, hd_PN2V, k=5):
    
    pre = pre_PN2V[:,:,:,0]+np.random.normal(scale=pre_PN2V[:,:,:,1])
    hd = hd_PN2V[:,:,:,0]+np.random.normal(scale=hd_PN2V[:,:,:,1])
    
    k=float(k)
    
    pre=np.nan_to_num(pre, copy=False)
    hd=np.nan_to_num(hd, copy=False)

    
    lowdose = pre + (hd-pre)/k

    normFactor = np.max(lowdose)
    
    ImagePre = pre/normFactor
    ImageLowDose = lowdose/normFactor
    ImageEnhancement=ImageLowDose-ImagePre

    return ImageEnhancement, ImageLowDose, normFactor