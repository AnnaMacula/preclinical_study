#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 12:33:23 2021

@author: user
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:52:31 2020

@author: bracco
"""
# import sys
import numpy as np
import os
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from contextlib import redirect_stdout

import tensorflow as tf

# from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Nadam
import datetime
import subprocess
import models
import callbacks
import DataGenerators
import io_model
import argparse
import losses
PATHS = io_model.configuration()



default_path = PATHS["path_trained_models"]


parser = argparse.ArgumentParser()

parser.add_argument("-GPU",help="select GPU, default: 0",  choices=["0", "1"], default="0")
parser.add_argument("-ns","--network_scheme",help="select network scheme, default: BN_RELU_CANONICAL", choices=["BN_RELU_CANONICAL", "RELU_BN_CANONICAL","RELU_BN_CUSTOM"], default="BN_RELU_CANONICAL")

group1 = parser.add_mutually_exclusive_group()
group1.add_argument("-ACQ","--acquired", help="if flagged ACQ set is selected", action="store_true")
group1.add_argument( "-nl", "--noise_level", help="select SIM dataset and noise level, e.g. -nl 0.015. Default: 0.0", default=0.0, type=float)

parser.add_argument("-mode", "--modality", help = "select modality: dose reduction/booster (dr), dose reduction/booster (dr_3D), virtual contrast (vc)", choices = ["dr","dr_3D", "vc"], default = "dr")
parser.add_argument("--lr_mode", "--lrmodality", help = "select learning rate modality: lr/decay (standard) or cosine restart (restart)", choices = ["standard", "restart"], default = "standard")



parser.add_argument("-lr", "--learning_rate", help="select learning rate, default: 0.01", type=float , default=0.01)
parser.add_argument("-d", "--decay", help="select decay, default: 0.001", type=float, default=0.001)

parser.add_argument("--steplr", "--learning_rate_step", help="select learning rate step, default: 1000", type=float , default=1000)
parser.add_argument("--tmul", "--learning_rate_tmul", help="select learning rate tmul, default: 2", type=float , default=2)
parser.add_argument("--mmul", "--learning_rate_mmul", help="select learning rate mmul, default: 1", type=float , default=1)

parser.add_argument("-id","--exp_id", help="choose exp id")

parser.add_argument("-src", "--data_source", help="select source, default: PRECLINIC",  
                    choices=["preclinic2020", "preclinic2021_2D", "preclinic2021_3D","clinic_cns", "clinic_liver", "clinic_ctcns", "clinic_vc"], default="preclinic2020")


group = parser.add_argument_group("coupling coefficients")
group.add_argument("-a", type=float, help='select pixelwise coupling, default: None', default=None)
group.add_argument("-b", type=float, help='select fft coupling,  default: None', default=None)
group.add_argument("-c", type=float, help='select perceptual coupling,  default: None', default=None)


parser.add_argument("-lpxl", "--pixelwise_loss", help="select pixelwise component, default: MAE", choices = ["MAE", "dWMAE"], default="MAE")
parser.add_argument("-lper", "--perceptual_loss", help="select perceptual component, default: VGG19", choices = ["VGG16", "VGG19"], default="VGG19")
parser.add_argument("-pl","--pooling_layers", help="select number of pooling layers, default: 4", choices=[3, 4, 5], default=4, type=int)
parser.add_argument("-eps", "--epochs", help="select number of epochs, default: 250", type=int, default=250)
parser.add_argument("-s", "--steps_per_epoch", help="select steps per epoch, default: 50", type=int, default=50)

parser.add_argument("-bs", "--batch_size", help="select batch_size, default: 20", type=int, default=20)

parser.add_argument("-vs", "--validation_steps", help="select validation steps, default: 10", type=int, default=10)
parser.add_argument("-r3d", "--range_3d", help="select 3D range, default: 0", type=int, choices=[1,2,3], default=0)


parser.add_argument("--chunk", help="number of images per patient, default: auto", nargs="+", type=str, default=["pre", "lowdose", "highdose"])#,  default=["auto"])#NOTA cambiato default prima era choices


parser.add_argument("-NN", "--no_norm", help="if flagged skips normalization",  action="store_true")

parser.add_argument("-sum", "--summary", help="if flagged displays network summary", action="store_true")
parser.add_argument("--output3d", help="if flagged network will produce 3d output", action="store_true")

args = parser.parse_args()

# If run through spyder
if 'SPY_PYTHONPATH' in os.environ:
    print("Script running in Spyder console.")    
    args.exp_id, args.c, args.clinic, args.batch_size, args.range_3d  = "test", 1.0, "cns", 20, 0
    args.data_source = "preclinic2021_2D"
    args.strategy = None
    args.kwargs = None
    args.lr_mode = "standard"
    args.noise_level = 0.015
    args.acquired = False
    # args.chunk = ["pre", "lowdose", "post"]


####CAMBIATO DOPO SPOSTAMENTO CODICE#######
args.strategy = None
args.kwargs = None    
####CAMBIATO DOPO SPOSTAMENTO CODICE#######


group_actions = vars(group)["_group_actions"]
group_dict = {vars(group_action)["dest"] : vars(args)[vars(group_action)["dest"]] for group_action in group_actions}


# check for parsing errors
if args.modality=="dr_3D":
    if (args.range_3d==1 and args.batch_size>8) or (args.range_3d==2 and args.batch_size>5) or (args.range_3d==3 and args.batch_size>3):
        parser.error('Batch size too big for available memory.')


#MORE PARSING
data_source = args.data_source
training_set = "ACQ" if args.acquired else "SIM"
train_folder = PATHS[f"path_train_{data_source.lower()}"] + f"/data_for_training_{training_set.lower()}_{args.GPU}"
train_path, validation_path = f"{train_folder}/train", f"{train_folder}/val"
GPU = args.GPU
chunk = args.chunk
nil = len(chunk) - 1


    
    
os.environ["CUDA_VISIBLE_DEVICES"] = GPU; 

def main():
     
    print("Initializing loss functions and metrics...")

    modelcut, name = io_model.modelcut_new(f"{args.perceptual_loss}-{args.pooling_layers}")
    loss_components = [args.pixelwise_loss, "FFT_MAE", losses.Perceptual(modelcut, name)]
    
    loss_functions, args_dictionary = io_model.parse_loss_new(*loss_components, normalize=not args.no_norm, 
                                                          data_source=data_source, **group_dict)
    
    
    LOSS_fun = loss_functions["composite_loss"]  
    args_dictionary["start"] = str(datetime.datetime.now())
    
    
    #LOAD MODEL + COMPILE
    if "3D" in args.modality:
        m = models.model_3d((1+2*args.range_3d,256,256,2), args.network_scheme, output3d=args.output3d)
    else:
        m = models.model((256,256,nil+4*args.range_3d), args.network_scheme)

    if args.lr_mode == "standard":
        #LEARNING RATE/DECAY ADAM
        print("Learning rate and decay modality selected")
        m.compile(loss= LOSS_fun, optimizer=Adam(learning_rate=args.learning_rate, decay=args.decay), metrics=['mean_squared_error'])
        
    elif args.lr_mode == "restart":
        #LEARNING RATE RESTART
        initial_learning_rate = args.learning_rate
        first_decay_steps = args.steplr   
        t_mul = args.tmul #fattore moltiplicativo di # step dopo il restart, es se tmul = 1 il numero di step rimane fisso, mmul = 2 il numero di step raddoppia ogni volta
        m_mul = args.mmul #fattore moltiplicativo di lr dopo il restart, es se mmul = 1 lr riparte dallo stesso valore, mmul = 2 riparte da volere doppio
        #alpha è il lr minimo lo teniamo fisso a zero
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha=0.0, name=None)
        print("Learning rate with restart modality selected")
        m.compile(loss= LOSS_fun, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['mean_squared_error'])

    
    
    # define the name of the directory to be created        
    path = PATHS["path_trained_models"] + f"/Composite_losses/{args.exp_id}"

    try:
        os.mkdir(path)
        print (f"Successfully created the directory {path}")
    except OSError:
        print (f"Creation of the directory {path} failed")
        
        
    # if summary is flagged displays and saves summary
    if args.summary:    
            m.summary()
    with open("model_summary.txt", "w") as f:
        with redirect_stdout(f):
            m.summary()
            
            
    #CALLBACKS
    checkPointWeightsFileName = f"SnapShot_{args.exp_id}"
    checkpoint = callbacks.Checkpoints(f"{path}/{checkPointWeightsFileName}.npy", verbose = 1, mode = 'min')       
    checkpoint_std = ModelCheckpoint(f"{path}/{args.exp_id}.h5", monitor = "val_loss", verbose = 1, save_best_only = True, 
                                     mode = "auto", period = 1)

    csv_logger = CSVLogger(f"{path}/log.out", append=False, separator=';')
    
    callbacks_list = [checkpoint, csv_logger, checkpoint_std]
    
    if args.modality ==  "dr":
        train_gen = DataGenerators.dataGenDoseReduction2_5D_lowRAM_new(args.batch_size, "train", args.range_3d, (256,256), 90, True, True, 
                                                                       args.noise_level, None, 50, args.steps_per_epoch, args.strategy, args.kwargs, data_source, GPU, training_set, chunk)
        validation_gen = DataGenerators.dataGenDoseReduction2_5D_lowRAM_new(args.batch_size, "val", args.range_3d, (256,256), 90, True, True,
                                                                             args.noise_level, None, 50, args.steps_per_epoch, args.strategy, args.kwargs, data_source, GPU, training_set, chunk)
    elif args.modality == "vc":
        train_gen = DataGenerators.dataGenDoseReduction2D_lowRAM_new_VC(args.batch_size, "train", (256,256), 90, True, True, 
                                                                        args.noise_level, None, 50, args.steps_per_epoch, args.strategy, args.kwargs, data_source, GPU, training_set, chunk)
        validation_gen = DataGenerators.dataGenDoseReduction2D_lowRAM_new_VC(args.batch_size, "val", (256,256), 90, True, True, 
                                                                             args.noise_level, None, 50, args.steps_per_epoch, args.strategy, args.kwargs, data_source, GPU, training_set, chunk)            
    elif "3D" in args.modality:        
        train_gen = DataGenerators.dataGenDoseReduction3D_lowRAM_new(args.batch_size, "train", (256,256), 90, True, True, 
                                                                        args.noise_level, None, 50, args.steps_per_epoch, args.strategy, args.kwargs, data_source, GPU, training_set, chunk)
        validation_gen = DataGenerators.dataGenDoseReduction3D_lowRAM_new(args.batch_size, "val", (256,256), 90, True, True, 
                                                                             args.noise_level, None, 50, args.steps_per_epoch, args.strategy, args.kwargs, data_source, GPU, training_set, chunk)            
 
        
    #TRAINING
    results = m.fit(train_gen, epochs = args.epochs, steps_per_epoch = args.steps_per_epoch, validation_data = validation_gen, 
                    validation_steps = args.validation_steps, callbacks = callbacks_list)    
    
    
    #PERFORMANCES AND RESULTS        
    #save the trained model as a .h5 file in the same labelled folder
    m.save(f"{path}/{args.exp_id}_last.h5")       
  
    # save history in two .npy files
    np.save(f"{path}/{args.exp_id}_train_loss_history.npy" , results.history["loss"])
    np.save(f"{path}/{args.exp_id}_val_loss_history.npy" , results.history["val_loss"])
    
    # complete dictionary
    with open(f"{train_folder}/current_random_seed.txt", "r") as f:
        random_seed = f.readline().split(sep=" ")[-1]
    
    args_dictionary["random_seed"] = random_seed
    args_dictionary["loss_function"] = "Network_and_pixelwise"
    args_dictionary["folder_path"] = path
    args_dictionary["finish"] = str(datetime.datetime.now())
    # args_dictionary["a"], args_dictionary["b"], args_dictionary["c"] = a, b, c
    
    with open(f"{path}/{args.exp_id}_training_parameters.txt","w") as f:
        json.dump({**args_dictionary, **vars(args)}, f)


    return results

if __name__ == "__main__":
    pippo = main()
