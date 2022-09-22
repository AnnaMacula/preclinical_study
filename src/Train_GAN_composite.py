#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:18:54 2021

@author: user
"""


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import losses
import GdDoseReductionModel
import DataGenerators
import io_model
import os, sys
# import numpy as np
import argparse
from tensorflow.keras import backend as K
import callbacks
import json


PATHS = io_model.configuration()
path = "/home/user/Scrivania/Preclinical Study/Trained_models/GANs"

if not os.path.exists(path):
    os.mkdir(path)

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--epochs", help="select number of epochs, default: 250", type=int, default=250)
parser.add_argument("-s", "--steps_per_epoch", help="select steps per epoch, default: 50", type=int, default=50)
parser.add_argument("-bs", "--batch_size", help="select batch_size, default: 20", type=int, default=20)
parser.add_argument("-vs", "--validation_steps", help="select validation steps, default: 10", type=int, default=10)

# group1 = parser.add_mutually_exclusive_group()
# group1.add_argument("-ACQ","--acquired", help="if flagged ACQ set is selected", action="store_true")
parser.add_argument( "-nl", "--noise_level", help="select SIM dataset and noise level, e.g. -nl 0.015. Default: None", default=None, type=float)

parser.add_argument("-gen_lr", help="select generator learning rate, default: 0.01", type=float , default=0.01)
parser.add_argument("-gen_d", help="select generator decay, default: 0.001", type=float, default=0.001)

parser.add_argument("-disc_lr", help="select discriminator learning rate, default: 0.01", type=float , default=0.001)
parser.add_argument("-disc_d", help="select discriminator decay, default: 0.001", type=float, default=0.0001)


parser.add_argument("-a", type=float, help='select pixelwise coupling, default: None', default=None)
parser.add_argument("-b", type=float, help='select fft coupling,  default: None', default=None)
parser.add_argument("-c", type=float, help='select perceptual coupling, default: None', default=None)
parser.add_argument("-d", type=float, help='select GAN coupling, default: 1.0', default=None)
parser.add_argument("-rf", "--red_factor", type=int, help='select reduction factor. Default: 8', default=8)

parser.add_argument("--rescale", help="if flagged activates rescaling",  action="store_true")

parser.add_argument("-id","--exp_id", help="choose exp id")


args = parser.parse_args()

training_set = "ACQ" if args.noise_level is None else "SIM"


epochs = args.epochs
batch_size = args.batch_size
steps_per_epoch = args.steps_per_epoch
validation_steps = args.validation_steps
exp_id = args.exp_id


# If run through spyder
if 'SPY_PYTHONPATH' in os.environ:
    print("Script running in Spyder console.")    
    epochs = 1
    exp_id = "test0_Rep1"
    args.d = 1.0

rescale = 10**6 if args.rescale else None

print("Initializing generator network...")
generator = GdDoseReductionModel.model(inputSize=(256, 256, 2), floatx = 'float32' )
print("Done.")

print("Initializing discriminator network...")
discriminator = losses.custom_VGG16(input_shape=(256, 256, 1), red_factor=args.red_factor, rescale=rescale)
print("Done.")


generator_optimizer, discriminator_optimizer = Adam(learning_rate=args.gen_lr, decay=args.gen_d), Adam(learning_rate=args.disc_lr, decay=args.disc_d)

print("Initializing loss functions and metrics...")
generator_loss = losses.composite_loss(network_label="VGG19", layer=4,  a=args.a, b=args.b, c=args.c, d=args.d)
discriminator_loss, discriminator_metric = losses.discriminator_loss, tf.keras.metrics.BinaryAccuracy()
print("Done.")

print("Initializing generators...")
# DataGenerator = DataGenerators.DataGen_GAN(batch_size, strategy=None, vol_size=(24, 256, 256), 
#                                            max_rot=90, flip_h=True, flip_v=True, noise_level=0.0, seed=None)
train_gen = DataGenerators.dataGenDoseReduction_new(batch_size, "train",  vol_size=(24,256,256), 
                                              max_rot=90, flip_h=True, flip_v=True, noise_level=args.noise_level, seed=None)
val_gen = DataGenerators.dataGenDoseReduction_new(batch_size, "val",  vol_size=(24,256,256), 
                                              max_rot=90, flip_h=True, flip_v=True, noise_level=args.noise_level, seed=None)
print("Done.")


if "test0" not in exp_id:
    k = 1
    while os.path.exists(f"{path}/{exp_id}"):    
        exp_id = exp_id.replace(f"Rep{k}", f"Rep{k+1}") 
        k += 1
   
if not os.path.exists(f"{path}/{exp_id}"):
    os.mkdir(f"{path}/{exp_id}")




@tf.function
# def train_step(batch_disc, batch_gen):
def train_step(batch):

    ins, out = K.cast(batch[0], tf.float32), K.cast(batch[1], tf.float32)


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(ins, training=True)

        real_output, fake_output = discriminator(out, training=True), discriminator(generated_images, training=True)
  
        # gen_loss = generator_loss(fake_output, K.cast(batch_gen[1], tf.float32), generated_images)
        gen_loss = generator_loss(fake_output, out, generated_images)

        disc_loss = discriminator_loss(real_output, fake_output)
        
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
        
        accuracy = discriminator_metric(y1, tf.concat([K.cast(fake_output, tf.float32), K.cast(real_output, tf.float32)], axis=0))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
    return gen_loss, disc_loss, accuracy


@tf.function
def val_step(batch):
    
    ins, out = K.cast(batch[0], tf.float32), K.cast(batch[1], tf.float32)
    # disc_in, disc_out = K.cast(batch_disc[0], tf.float32), K.cast(batch_disc[1], tf.float32)

    
    generated_images = generator(ins, training=False)
    
    real_output, fake_output = discriminator(out, training=False), discriminator(generated_images, training=False)
    
    # gen_loss = generator_loss(fake_output, K.cast(batch_gen[1], tf.float32), generated_images)
    gen_loss = generator_loss(fake_output, out, generated_images)

    disc_loss = discriminator_loss(real_output, fake_output)
    
    y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
    
    accuracy = discriminator_metric(y1, tf.concat([K.cast(fake_output, tf.float32), K.cast(real_output, tf.float32)], axis=0))

    return gen_loss, disc_loss, accuracy


    
def train_GAN(generator, discriminator, batch_size, epochs=50, steps_per_epoch=50, validation_steps=10):
    
    checkpoint = callbacks.GANpoint(f"{path}/{exp_id}/{exp_id}.h5", monitor="gen_val_loss", verbose = 1, mode = "auto")
    checkpoint.on_train_begin(epochs, steps_per_epoch, validation_steps)
     
    for epoch in checkpoint.t_epochs:
 
        checkpoint.on_epoch_begin(epoch)
           
        for step in checkpoint.t_steps:
            
            # batch_disc, batch_gen = DataGenerator.current_train
            train_batch = next(train_gen)
            step_results = train_step(train_batch)
            
            checkpoint.on_step_end("train", step_results) 
            
            # DataGenerator.next_train()
        
        for step in checkpoint.v_steps:
              
            # batch_val_disc, batch_val_gen = DataGenerator.current_val
            val_batch = next(val_gen)
            
            step_results = val_step(val_batch)
            
            checkpoint.on_step_end("val", step_results) 
            
            # DataGenerator.next_val()
        
        checkpoint.set_models(generator, discriminator)       
        checkpoint.on_epoch_end(epoch, logs=checkpoint.logs)        

    dic = {}       
    dic["generator"], dic["discriminator"] = generator, discriminator
    dic["history"] = checkpoint.logs
    
    checkpoint.on_train_end(f"{path}/{exp_id}")
    
    return dic
        

print("Initiating training")
results = train_GAN(generator, discriminator, batch_size, epochs, steps_per_epoch, validation_steps)

# history = results["history"]

# print("Saving results...")
# results["discriminator"].save(f"{path}/{exp_id}/discriminator_last.h5")
# results["generator"].save(f"{path}/{exp_id}/generator_last.h5")
# print("Done.")


# save VGG loss and accuracy
# print("Saving history...")

# os.mkdir(f"{path}/{exp_id}/logs")

# for key in history.keys():
#     np.save(f"{path}/{exp_id}/logs/{key}.npy", np.array(history[f"{key}"]))
# print("Done.")


print("Saving training parameters...")
with open(f"{path}/{exp_id}/{exp_id}_training_parameters.txt","w") as f:
    json.dump(vars(args), f)
print("Done.")

print(f"Training of model {exp_id} complete.")

