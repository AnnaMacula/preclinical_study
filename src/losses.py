#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:14:20 2022

@author: user
"""
import tensorflow as tf
from tensorflow.keras import backend as K

import decorators

@decorators.make_compatible()
def MAE(y_true, y_pred):
    
    mae_on_batch = K.abs(y_true - y_pred)        
    mae = K.mean(mae_on_batch)
    
    return mae
    
@decorators.make_compatible()
def FFT_MAE(y_true, y_pred):
    
    cast_true = K.cast(y_true[:,:,:,0], tf.complex128)
    cast_pred = K.cast(y_pred[:,:,:,0], tf.complex128)

    fft_true = tf.signal.fft2d(cast_true)
    fft_pred = tf.signal.fft2d(cast_pred)
           
    fft_mae = K.mean(K.abs(fft_true - fft_pred))

    return  fft_mae

def Perceptual(modelnetwork, name, channels=3):
    
    @decorators.make_compatible()
    def loss(y_true, y_pred):
        
        y_true = tf.repeat([y_true[...,0]], repeats=channels, axis=0)
        y_pred = tf.repeat([y_pred[...,0]], repeats=channels, axis=0)

        y_true = K.permute_dimensions(y_true, (1,2,3,0))       
        y_pred = K.permute_dimensions(y_pred, (1,2,3,0))       
    
        y_true_network = modelnetwork(y_true)
        y_pred_network = modelnetwork(y_pred)
        
        network_loss = K.square(y_true_network - y_pred_network)
    
        network_mean = K.mean(network_loss)
    
        return network_mean
        
    loss.__name__ = name
    
    return loss



def Perceptual_3d(modelnetwork, name, range3D=0):
    
    
    @decorators.make_compatible()
    def loss(y_true, y_pred):
        
        y_true = K.stack((y_true[...,0], y_true[...,0], y_true[...,0]), axis=-1)  #triplicare il tensore in shape (shape(y),3)
        y_pred = K.stack((y_pred[...,0], y_pred[...,0], y_pred[...,0]), axis=-1)

        y_true_network = modelnetwork(tf.reshape(y_true, [-1, 256, 256, 3]))
        y_pred_network = modelnetwork(tf.reshape(y_pred, [-1, 256, 256, 3]))

        network_loss = K.square(y_true_network - y_pred_network)
    
        network_mean = K.mean(network_loss)
    
        return network_mean
    
    loss.__name__ = name

    return loss


def GAN_pix2pix(from_logits=True, args_positions=(0,1,2)):

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    @decorators.make_compatible(args_positions)
    def generator_loss(target, gen_output, disc_generated_output):
        
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        return gan_loss
    
    generator_loss.__name__ = "GAN_pix2pix"
    
    
    def discriminator_loss(disc_real_output, disc_generated_output):
        
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
        
    return generator_loss, discriminator_loss


def standard_loss_composition(dic, floatx=tf.float64):#, **kwargs):
    
    
    components = {}
    for key, [func, value] in dic.items():

        def component(value, func):
            
            return lambda *args : value*K.cast(func(*args), floatx)
            
        components[f"{key}_loss"] = component(value, func)
        
    
    def loss(*args):
        
        lsum = 0.0
        for component in components.values():   
            lsum = lsum + component(*args)
            
        return lsum
    
    return {**{"composite_loss" : loss}, **components}






