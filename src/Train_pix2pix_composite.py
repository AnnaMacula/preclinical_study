#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:18:54 2021

@author: user
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import losses
# import GdDoseReductionModel
import DataGenerators
import io_model, models
import os, sys, shutil
# import numpy as np
import argparse
from tensorflow.keras import backend as K
import callbacks
import json
import datetime

PATHS = io_model.configuration()
path = "/home/user/Scrivania/Preclinical Study/Trained_models/GANs"

if not os.path.exists(path):
    os.mkdir(path)

parser = argparse.ArgumentParser()

group1 = parser.add_mutually_exclusive_group()
group1.add_argument("-ACQ","--acquired", help="if flagged ACQ set is selected", action="store_true")
group1.add_argument( "-nl", "--noise_level", help="select SIM dataset and noise level, e.g. -nl 0.015. Default: 0.0", default=0.0, type=float)


parser.add_argument("-eps", "--epochs", help="select number of epochs, default: 250", type=int, default=250)
parser.add_argument("-ts", "--steps_per_epoch", help="select steps per epoch, default: 50", type=int, default=50)
parser.add_argument("-bs", "--batch_size", help="select batch_size, default: 20", type=int, default=20)
parser.add_argument("-vs", "--validation_steps", help="select validation steps, default: 10", type=int, default=10)

parser.add_argument("-gen_lr", help="select generator learning rate, default: 0.01", type=float , default=0.01)
parser.add_argument("-gen_d", help="select generator decay, default: 0.001", type=float, default=0.001)


parser.add_argument("-src", "--data_source", help="select source, default: PRECLINIC",  
                    choices=["preclinic2020", "preclinic2021_2D", "preclinic2021_3D","clinic_cns", "clinic_liver", "clinic_ctcns", "clinic_vc"], default="preclinic2020")

# parser.add_argument("-disc_lr", help="select discriminator learning rate, default: 0.01", type=float , default=0.001)
# parser.add_argument("-disc_d", help="select discriminator decay, default: 0.001", type=float, default=0.0001)

group = parser.add_argument_group("coupling coefficients")
group.add_argument("-a", type=float, help='select pixelwise coupling, default: None', default=None)
group.add_argument("-b", type=float, help='select fft coupling,  default: None', default=None)
group.add_argument("-c", type=float, help='select perceptual coupling, default: None', default=None)
group.add_argument("-d", type=float, help='select GAN coupling, default: None', default=None)

parser.add_argument("-NN", "--no_norm", help="if flagged skips normalization",  action="store_true")
parser.add_argument("--chunk", help="number of images per patient, default: auto", nargs="+", type=str, 
                    default=["pre", "lowdose", "highdose"])#,  default=["auto"])#NOTA cambiato default prima era choices

parser.add_argument("-r3d", "--range_3d", help="select 3D range, default: 0", type=int, choices=[1,2,3], default=0)

parser.add_argument("-id","--exp_id", help="choose exp id")


args = parser.parse_args()





epochs = args.epochs
batch_size = args.batch_size
steps_per_epoch = args.steps_per_epoch
validation_steps = args.validation_steps
exp_id = args.exp_id

training_set = "ACQ" if args.acquired else "SIM"



# If run through spyder
if 'SPY_PYTHONPATH' in os.environ:
    print("Script running in Spyder console.")    
    epochs = 10
    exp_id = "test0"
    args.a = 1.0
    args.b = 1.0
    args.c = 1.0
    args.d = 1.0
    if os.path.exists(f"{path}/{exp_id}"):
        shutil.rmtree(f"{path}/{exp_id}")



group_actions = vars(group)["_group_actions"]
group_dict = {vars(group_action)["dest"]:vars(args)[vars(group_action)["dest"]] for group_action in group_actions}


if not os.path.exists(f"{path}/{exp_id}"):
    os.mkdir(f"{path}/{exp_id}")


print("Initializing generator network...")
generator = models.model(inputSize=(256, 256, 2), floatx = 'float32' )
print("Done.")

print("Initializing discriminator network...")
discriminator = models.discriminator_pix2pix(inputShape=[256, 256, 2], targetShape=[256, 256, 1])
print("Done.")


gen_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=args.gen_lr, first_decay_steps=1000, t_mul=2.0, m_mul=1.0, alpha=0.0,
    name=None)

disc_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=2e-4, first_decay_steps=1000, t_mul=2.0, m_mul=1.0, alpha=0.0,
    name=None)

generator_optimizer, discriminator_optimizer = Adam(2e-4, beta_1=0.5), Adam(2e-4, beta_1=0.5)

# generator_optimizer, discriminator_optimizer = Adam(learning_rate=gen_scheduler), Adam(learning_rate=disc_scheduler)

print("Initializing loss functions and metrics...")
generator_loss, discriminator_loss = losses.GAN_pix2pix()

loss_components = ["MAE", "FFT_MAE", losses.Perceptual(*io_model.modelcut_new("VGG19-4")), generator_loss]




loss_functions, additional_info = io_model.parse_loss_new(*loss_components, normalize=not args.no_norm, 
                                                          data_source=args.data_source, **group_dict)


discriminator_metric = tf.keras.metrics.BinaryAccuracy()
print("Done.")

print("Initializing generators...")
train_gen = DataGenerators.dataGenDoseReduction2_5D_lowRAM_new(args.batch_size, "train", args.range_3d, (256,256), 90, True, True, 
                                                                       args.noise_level, None, 50, args.steps_per_epoch, None, None, args.data_source, "0", training_set, args.chunk)
val_gen = DataGenerators.dataGenDoseReduction2_5D_lowRAM_new(args.batch_size, "val", args.range_3d, (256,256), 90, True, True,
                                                                             args.noise_level, None, 50, args.steps_per_epoch, None, None, args.data_source, "0", training_set, args.chunk)

# train_gen = DataGenerators.dataGenDoseReduction_new(batch_size, "train",  vol_size=(24,256,256), 
#                                               max_rot = 90, flip_h = True, flip_v = True, noise_level= None, seed = None)
# val_gen = DataGenerators.dataGenDoseReduction_new(batch_size, "val",  vol_size=(24,256,256), 
#                                               max_rot = 90, flip_h = True, flip_v = True, noise_level= None, seed = None)
print("Done.")


summary_writer = tf.summary.create_file_writer(f"{path}/{exp_id}/tensorboard/fit/"
                                               + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_dir = f"{path}/{exp_id}/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                 discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)



@tf.function
def train_step(batch, epoch):

    ins, out = K.cast(batch[0], tf.float32), K.cast(batch[1], tf.float32)


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(ins, training=True)

        disc_real_output = discriminator([ins, out], training=True)
        disc_generated_output = discriminator([ins, gen_output], training=True)
          
        results = {key : loss(out, gen_output, disc_generated_output) for key, loss in loss_functions.items()}
        
        # gen_loss = results["composite_loss"]
        
        results["disc_loss"] = discriminator_loss(disc_real_output, disc_generated_output)
 
        # disc_loss = results["disc_loss"]

        
        y1 = tf.concat([tf.zeros_like(disc_generated_output), tf.ones_like(disc_real_output)], axis=0)
        
        results["accuracy"]  = discriminator_metric(y1, tf.concat([K.cast(disc_generated_output, tf.float32),
                                                       K.cast(disc_real_output, tf.float32)], axis=0))
        
    generator_gradients = gen_tape.gradient(results["composite_loss"],
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(results["disc_loss"],
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    with summary_writer.as_default():
        for key, result in results.items():         
            tf.summary.scalar(key, result, step=epoch)
  
        # tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        # tf.summary.scalar('accuracy', accuracy, step=epoch)

    return results

@tf.function
def val_step(batch, epoch):
    
    ins, out = K.cast(batch[0], tf.float32), K.cast(batch[1], tf.float32)
    
    gen_output = generator(ins, training=False)
    
    disc_real_output = discriminator([ins, out], training=False)
    disc_generated_output = discriminator([ins, gen_output], training=False)
    
    results = {key : loss(out, gen_output, disc_generated_output) for key, loss in loss_functions.items()}
        
   
    results["disc_loss"] = discriminator_loss(disc_real_output, disc_generated_output)
    
    
    y1 = tf.concat([tf.zeros_like(disc_generated_output), tf.ones_like(disc_real_output)], axis=0)
        
    results["accuracy"] = discriminator_metric(y1, tf.concat([K.cast(disc_generated_output, tf.float32),
                                                       K.cast(disc_real_output, tf.float32)], axis=0))
    
    with summary_writer.as_default():
        for key, result in results.items():         
            tf.summary.scalar(key, result, step=epoch)
  
    return results



def train_pix2pix(generator, discriminator, batch_size, epochs=50, steps_per_epoch=50, validation_steps=10, chkp_freq=50):
    
    callback = callbacks.GANpoint_new(f"{path}/{exp_id}/{exp_id}.h5", monitor="val_loss", verbose = 1, mode = "auto",
                                      epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                       logs_entries=[*loss_functions.keys(), "disc_loss", "accuracy"])
    
    callback.on_train_begin(keys_to_monitor = ["composite_loss", "disc_loss", "accuracy"])
     
    for epoch in callback.t_epochs:
 
        callback.on_epoch_begin(epoch)
           
        for step in callback.tqdm_bars["train"]:
            
            callback.on_step_begin(step)
            
            train_batch = next(train_gen)
            step_results = train_step(train_batch, epoch)
            
            callback.on_step_end("train", step_results) 
        
        for step in callback.tqdm_bars["val"]:
            
            callback.on_step_begin(step)
            
            val_batch = next(val_gen)            
            step_results = val_step(val_batch, epoch)
            
            callback.on_step_end("val", step_results) 
            
        if (epoch + 1) % chkp_freq==0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        callback.set_models(generator, discriminator)       
        callback.on_epoch_end()        

    dic = {}       
    dic["generator"], dic["discriminator"] = generator, discriminator
    dic["history"] = callback.logs
    
    callback.on_train_end(f"{path}/{exp_id}")
    
    return dic
        

print("Initiating training")
results = train_pix2pix(generator, discriminator, batch_size, epochs, steps_per_epoch, validation_steps, chkp_freq=125)


print("Saving training parameters...")
with open(f"{path}/{exp_id}/{exp_id}_training_parameters.txt","w") as f:    
    json.dump({**vars(args), **additional_info}, f)
    
with open(f"{path}/{args.exp_id}_lr_schedule.txt","w") as f:
    json.dump(tf.keras.optimizers.schedules.serialize(gen_scheduler), f)
print("Done.")


print(f"Training of model {exp_id} complete.")
