#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:46:21 2020

@author: bracco
"""

import sys
import numpy as np
import logging
import os
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
import math
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import itertools

class Checkpoints(Callback):
    
    def __init__(self, filepath, monitor='val_loss', verbose=0, mode='auto'):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.verbose = verbose
        
        if mode not in ['auto', 'min', 'max']:
            logging.warning(f'ModelCheckpoint mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'          
          
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
            
    def set_model(self, model):
        self.model = model
      
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f'Can save best model only with {self.monitor} available, ','skipping.')
        else:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f'\n Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current},', f' saving model to {self.filepath}')
                    self.best = current
                    # self.model.save(self.filepath)
                    
                    np.save(self.filepath, self.model.get_weights()) 
            else:
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}' )        
    
                    
    
    
class LearningRateHistory(Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        lr_l = K.eval(self.model.optimizer.lr)
        lr_l *= (1. / (1. + K.eval(self.model.optimizer.decay) * K.eval(self.model.optimizer.iterations)))
        print(f"learng rate: {lr_l}")
 

        
        
class GANpoint(Callback):
    
    def __init__(self, filepath, monitor='val_loss', verbose=0, mode='auto', gen_optimizer=None, disc_optimizer=None):
        
        super(Callback, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.verbose = verbose
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        self.logs = {}
        
        if mode not in ['auto', 'min', 'max']:
            logging.warning(f'ModelCheckpoint mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'          
          
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
            
    def set_models(self, generator, discriminator):
        
        self.generator = generator
        self.discriminator = discriminator

    def on_epoch_end(self, epoch, logs={}):
         
        self.logs["gen_loss"].append(np.mean(self.gen_loss))
        self.logs["disc_loss"].append(np.mean(self.disc_loss))
        self.logs["accuracy"].append(np.mean(self.accuracy))
        self.logs["gen_val_loss"].append(np.mean(self.gen_val_loss))
        self.logs["disc_val_loss"].append(np.mean(self.disc_val_loss))
        self.logs["val_accuracy"].append(np.mean(self.val_accuracy))
        
        logs = logs or self.logs
        
        current = logs.get(self.monitor)
      
        current = current[-1] if isinstance(current, (list, dict, tuple, np.ndarray)) else current
        
        if self.monitor_op(current, self.best):                
            print(f'\n Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current},', f' saving model to {self.filepath}')
            self.best = current
            self.generator.save(self.filepath)
            
        self.t_epochs.set_postfix(gen=np.mean(self.gen_loss), disc=np.mean(self.disc_loss), acc=np.mean(self.accuracy))
   
    def on_train_begin(self, epochs, steps_per_epoch, validation_steps):
            
        self.logs["gen_loss"], self.logs["disc_loss"], self.logs["accuracy"] = [], [], []
        self.logs["gen_val_loss"], self.logs["disc_val_loss"], self.logs["val_accuracy"] = [], [], []
        
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        
        self.t_epochs = trange(epochs, desc="epoch", unit="epoch", postfix="loss = {loss:.8f}, accuracy = {accuracy:.4f}")    
        self.t_epochs.set_postfix(loss=0, accuracy=0)
        
        

    
    def on_epoch_begin(self, epoch):
                
        self.gen_loss, self.disc_loss, self.accuracy = [], [], []
        self.gen_val_loss, self.disc_val_loss, self.val_accuracy = [], [], []
        
        self.t_steps = tqdm(range(self.steps_per_epoch), desc="train", unit="batch", position=1, leave=False)
        self.v_steps = tqdm(range(self.validation_steps), desc="val", unit="batch", position=1, leave=False)

        
    def on_step_end(self, mode, results):
        
        step_gen_loss, step_disc_loss, step_accuracy = np.array(results)
        
        if mode=="train":
            self.gen_loss.append(step_gen_loss)
            self.disc_loss.append(step_disc_loss)
            self.accuracy.append(step_accuracy)
            
            self.t_steps.set_postfix(gen=step_gen_loss, disc=step_disc_loss, acc=step_accuracy)
            
        elif mode=="val":
            self.gen_val_loss.append(step_gen_loss)
            self.disc_val_loss.append(step_disc_loss)
            self.val_accuracy.append(step_accuracy)
                        
            self.v_steps.set_postfix(gen=step_gen_loss, disc=step_disc_loss, acc=step_accuracy)
    
    
    def reset_lr(self, optimizer, new_learning_rate):
        
        self.optimizer.lr.assign(new_learning_rate)
    
    def on_train_end(self, path):
        
        print("Saving results...")
        self.generator.save(f"{path}/generator.h5")
        self.discriminator.save(f"{path}/discriminator.h5")        
        print("Done.")
     
        # save generator loss, discriminator loss and accuracy
        print("Saving history...")        
        os.mkdir(f"{path}/logs")        
        for key in self.logs.keys():
            np.save(f"{path}/logs/{key}.npy", np.array(self.logs[f"{key}"]))
        print("Done.")
        



class GANpoint_new(Callback):
    
    def __init__(self, filepath, logs_entries, epochs, steps_per_epoch, validation_steps, 
                 monitor='val_loss', verbose=0, mode='auto', gen_optimizer=None, disc_optimizer=None):
        
        super(Callback, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.verbose = verbose
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        
        self.logs_entries = logs_entries
        self.logs = {"train" : {}, "val" : {}}
        self.modes = list(self.logs.keys())
        self.epochs = epochs
        self.mode_steps = {"train" : steps_per_epoch, "val" : validation_steps}

        self.epochs = epochs     

        if mode not in ['auto', 'min', 'max']:
            logging.warning(f'ModelCheckpoint mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'          
          
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
            
    def set_models(self, generator, discriminator):
        
        self.generator = generator
        self.discriminator = discriminator

    def on_train_begin(self, keys_to_monitor):
        
        for mode in self.modes:
            self.logs[mode] = {key : np.zeros(self.epochs) for key in self.logs_entries}
        
        self.t_epochs = trange(self.epochs, desc="epoch", unit="epoch", postfix="loss = {loss:.8f}, accuracy = {accuracy:.4f}")    
        self.t_epochs.set_postfix(loss=0, accuracy=0)
        self.keys_to_monitor = keys_to_monitor
    def on_epoch_begin(self, epoch):
        
        self.current_epoch = epoch
        
        self.epoch_results = {mode : {key : np.zeros(self.mode_steps[mode]) 
                                      for key in self.logs_entries} for mode in self.modes}
        
        self.tqdm_bars = {mode : tqdm(range(self.mode_steps[mode]), desc=mode, unit="batch", position=1, leave=False)
                                      for mode in self.modes}    
        
    def on_epoch_end(self):
        
        epoch = self.current_epoch
        
        for key, mode in itertools.product(self.logs_entries, self.modes):
            self.logs[mode][key] = np.array(self.epoch_results[mode][key]).mean()
        
        if self.monitor in ["auto", "val_loss"]:
            current = self.logs["val"]["composite_loss"]
        else:
            current = self.logs["val"][self.monitor]
        
        if self.monitor_op(current, self.best):                
            print(f'\n Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current},', f' saving model to {self.filepath}')
            self.best = current
            self.generator.save(self.filepath)
        
        key1, key2, key3 = [self.logs["train"][key] for key in self.keys_to_monitor]
    
        self.t_epochs.set_postfix(gen=key1, disc=key2, metric=key3)
        
    def on_step_begin(self, step):
        
        self.current_step = step
        
    def on_step_end(self, mode, results):
        
        for key in self.logs_entries:
            self.epoch_results[mode][key][self.current_step] = np.array(results[key])
        
        key1, key2, key3 = [np.array(results[key]) for key in self.keys_to_monitor]
        
        self.tqdm_bars[mode].set_postfix(gen=key1, disc=key2, metric=key3)
    
    def reset_lr(self, optimizer, new_learning_rate):
        
        self.optimizer.lr.assign(new_learning_rate)
    
    def on_train_end(self, path):
        
        print("Saving results...")
        self.generator.save(f"{path}/generator.h5")
        self.discriminator.save(f"{path}/discriminator.h5")        
        print("Done.")
     
        # save generator loss, discriminator loss and accuracy
        print("Saving history...")        
        os.mkdir(f"{path}/logs")
        
        metrics_keys = [key for key in self.logs_entries if "loss" not in key]
        losses_keys = [key for key in self.logs_entries if "loss" in key]
        
        for key in losses_keys:
            np.save(f"{path}/logs/{key}.npy", np.array(self.logs["train"][key]))
            np.save(f"{path}/logs/{key.replace('loss', 'val_loss')}.npy", np.array(self.logs["val"][key]))
            
        for key in metrics_keys:
            np.save(f"{path}/logs/{key}.npy", np.array(self.logs["train"][key]))
            np.save(f"{path}/logs/val_{key}.npy", np.array(self.logs["val"][key]))

        
        print("Done.")
        

        


        