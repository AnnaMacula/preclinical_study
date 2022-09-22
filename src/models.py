#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:26:55 2019

@author: user
"""



import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, Input, Add, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Lambda, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, MaxPool2D, Dense
tf.keras.backend.set_floatx('float32')

# Questa dovrebbe essere una variazione all'implementazione di: "Deep Learning Enables Reduced
# Gadolinium Dose for Contrast-Enhanced Brain MRI"
# Nota che in inputShortCut uso un kernel 3x3 mentre l'implementazione canonica prevede l'uso di un kernel 1x1
def Conv33_RELU_BN_Residuals_Add_Custom(inputLayer, numFeaturesMap, name = ""):
    
    
    conv1 = Conv2D(numFeaturesMap, (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = name + "_3x3Conv_RELU1")(inputLayer)
    batch_norm_1 = BatchNormalization(axis = -1, name = name + "_BN1")(conv1)
    
    conv2 = Conv2D(numFeaturesMap, (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = name + "_3x3Conv_RELU2")(batch_norm_1)
    batch_norm_2 = BatchNormalization(axis = -1, name = name + "_BN2")(conv2)

    inputShortCut = Conv2D(numFeaturesMap, (3,3), activation = 'relu', padding = 'same', 
                           kernel_initializer = 'he_normal', name = name + "_3x3ShortCut")(inputLayer)
    
    batch_norm_sc = BatchNormalization(axis = -1, name = name + "_BNShortCut")(inputShortCut)
    
    x = Add()([batch_norm_sc, batch_norm_2])
    
    x = Activation('relu')(x)
    
    x = BatchNormalization(axis = -1, name = name + "_BNfinal")(x)   
        
    return x


# Questa dovrebbe essere l'implementazione di: "Deep Learning Enables Reduced
# Gadolinium Dose for Contrast-Enhanced Brain MRI"
# Nota che in inputShortCut uso un kernel 1x1. Implementazione canonica.
def Conv33_RELU_BN_Residuals_Add_Canonical(inputLayer, numFeaturesMap, name = ""):
    
    conv1 = Conv2D(numFeaturesMap, (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = name + "_3x3Conv_RELU1")(inputLayer)
    batch_norm_1 = BatchNormalization(axis = -1, name = name + "_BN1")(conv1)
    
    if inputLayer.shape[3] < numFeaturesMap:
        numFeaturesMap = numFeaturesMap*2
    
    conv2 = Conv2D(numFeaturesMap, (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = name + "_3x3Conv_RELU2")(batch_norm_1)
    batch_norm_2 = BatchNormalization(axis = -1, name = name + "_BN2")(conv2)

    inputShortCut = Conv2D(numFeaturesMap, (1,1), activation = 'relu', padding = 'same', 
                           kernel_initializer = 'he_normal', name = name + "_1x1ShortCut")(inputLayer)
    
    batch_norm_sc = BatchNormalization(axis = -1, name = name + "_BNShortCut")(inputShortCut)
    
    x = Add()([batch_norm_sc, batch_norm_2])
    
    x = Activation('relu')(x)
    
    x = BatchNormalization(axis = -1, name = name + "_BNfinal")(x)   
        
    return x



# Qeusta andrebbe aggiornata seguendo lo schema in Conv33_RELU_BN_Residuals_Add
def Conv33_BN_RELU_Residuals_Add_Canonical(inputLayer, numFeaturesMap, name = ""):
    
    conv1 = Conv2D(numFeaturesMap, (3,3), activation = None, padding = 'same', 
                   kernel_initializer = 'he_normal', name = name + "_3x3Conv_1")(inputLayer)
    batch_norm_1 = BatchNormalization(axis = -1, name = name + "_BN1")(conv1)
    
    batch_norm_relu_1 = Activation('relu')(batch_norm_1)
    
    if inputLayer.shape[3] < numFeaturesMap:
        numFeaturesMap = numFeaturesMap*2
    
    conv2 = Conv2D(numFeaturesMap, (3,3), activation = None, padding = 'same', 
                   kernel_initializer = 'he_normal', name = name + "_3x3Conv_2")(batch_norm_relu_1)
    batch_norm_2 = BatchNormalization(axis = -1, name = name + "_BN2")(conv2)
    
    batch_norm_relu_2 = Activation('relu')(batch_norm_2)

    inputShortCut = Conv2D(numFeaturesMap, (1,1), activation = None, padding = 'same', 
                           kernel_initializer = 'he_normal', name = name + "_1x1ShortCut")(inputLayer)
    
    batch_norm_sc = BatchNormalization(axis = -1, name = name + "_BNShortCut")(inputShortCut)
    
    x = Add()([batch_norm_sc, batch_norm_relu_2])
    
    x = Activation('relu')(x)
    
    x = BatchNormalization(axis = -1, name = name + "_BNfinal")(x)   
        
    return x

# Scheme: one of  "BN_RELU_CANONICAL","RELU_BN_CANONICAL", "RELU_BN_CUSTOM".
def model(inputSize = (256,256,2), scheme = "BN_RELU_CANONICAL", floatx = 'float32' ):
    tf.keras.backend.set_floatx(floatx)

    inputs = Input(inputSize)
    
    pre_image_layer = Lambda(lambda x: x[:,:,:,0][...,None])(inputs)
    
    if scheme == "BN_RELU_CANONICAL":
        Conv33_RELU_BN_Residuals_Add = Conv33_BN_RELU_Residuals_Add_Canonical
    elif scheme == "RELU_BN_CANONICAL":
        Conv33_RELU_BN_Residuals_Add = Conv33_RELU_BN_Residuals_Add_Canonical
    elif scheme == "RELU_BN_CUSTOM":
        Conv33_RELU_BN_Residuals_Add = Conv33_RELU_BN_Residuals_Add_Custom
    
    Enc_1 = Conv33_RELU_BN_Residuals_Add(inputs, 24, "Encoder1")
    Enc_1_MP = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(Enc_1)
    
    Enc_2  = Conv33_RELU_BN_Residuals_Add(Enc_1_MP, 48, "Encoder2")
    Enc_2_MP = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(Enc_2)
    
    Enc_3  = Conv33_RELU_BN_Residuals_Add(Enc_2_MP, 96, "Encoder3")
    Enc_3_MP = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(Enc_3)
    
    Enc_4 = Conv33_RELU_BN_Residuals_Add(Enc_3_MP, 192, "Encoder4")
    Enc_4_MP = UpSampling2D(size=(2, 2))(Enc_4)
    
    Dec_1 = Conv33_RELU_BN_Residuals_Add(concatenate([Enc_3,Enc_4_MP]), 96, "Decoder1")
    Dec_1_US = UpSampling2D(size=(2, 2))(Dec_1)
    
    Dec_2 = Conv33_RELU_BN_Residuals_Add(concatenate([Enc_2,Dec_1_US]), 48, "Decoder2")
    Dec_2_US = UpSampling2D(size=(2, 2))(Dec_2)
    
    Dec_3 = Conv33_RELU_BN_Residuals_Add(concatenate([Enc_1,Dec_2_US]), 24, "Decoder3")
    
    OneOne_Conv = Conv2D(1, (1,1), activation=None)(Dec_3)
    #
    output = Add()([pre_image_layer, OneOne_Conv])
    
    model = Model(inputs = inputs, outputs = output)
        
    return model





# Qeusta andrebbe aggiornata seguendo lo schema in Conv33_RELU_BN_Residuals_Add
def Conv33_BN_RELU_Residuals_Add_Canonical_3d(inputLayer, numFeaturesMap, range3d, name = ""):
    
    
    conv1 = Conv3D(numFeaturesMap, (range3d+1,3,3), activation = None, padding = 'same', 
                   kernel_initializer = 'he_normal', name = f"{name}_{range3d}x3x3Conv_1")(inputLayer)
    batch_norm_1 = BatchNormalization(axis = -1, name = f"{name}_BN1")(conv1)
    
    batch_norm_relu_1 = Activation('relu')(batch_norm_1)
    
    if inputLayer.shape[4] < numFeaturesMap:
        numFeaturesMap = numFeaturesMap*2
    
    conv2 = Conv3D(numFeaturesMap, (range3d+1,3,3), activation = None, padding = 'same', 
                   kernel_initializer = 'he_normal', name = f"{name}_{range3d}x3x3Conv_2")(batch_norm_relu_1)
    batch_norm_2 = BatchNormalization(axis = -1, name = f"{name}_BN2")(conv2)
    
    batch_norm_relu_2 = Activation('relu')(batch_norm_2)

    inputShortCut = Conv3D(numFeaturesMap, (range3d+1,1,1), activation = None, padding = 'same', 
                           kernel_initializer = 'he_normal', name = f"{name}_{range3d}x1x1ShortCut")(inputLayer)
    
    batch_norm_sc = BatchNormalization(axis = -1, name = f"{name}_BNShortCut")(inputShortCut)
    
    x = Add()([batch_norm_sc, batch_norm_relu_2])
    
    x = Activation('relu')(x)
    
    x = BatchNormalization(axis = -1, name = name + "_BNfinal")(x)   
        
    return x



def model_3d(inputSize = (1,256,256,2), scheme = "BN_RELU_CANONICAL", floatx = "float32", output3d=False):
    tf.keras.backend.set_floatx(floatx)

    inputs = Input(inputSize)
    pre_image_layer = Lambda(lambda x: x[:,:,:,:,0][...,None])(inputs)
    
    range3d = int(inputSize[0]/2)
    
    Conv33_RELU_BN_Residuals_Add = Conv33_BN_RELU_Residuals_Add_Canonical_3d
       
    Enc_1 = Conv33_RELU_BN_Residuals_Add(inputs, 24, range3d=range3d, name = "Encoder1")
    Enc_1_MP = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding="same")(Enc_1)
    
    Enc_2  = Conv33_RELU_BN_Residuals_Add(Enc_1_MP, 48, range3d=range3d, name = "Encoder2")
    Enc_2_MP = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding="same")(Enc_2)
    
    Enc_3  = Conv33_RELU_BN_Residuals_Add(Enc_2_MP, 96, range3d=range3d, name = "Encoder3")
    Enc_3_MP = MaxPooling3D(pool_size=(inputSize[0], 2, 2), strides=(1, 2, 2), padding="same")(Enc_3)
    
    Enc_4 = Conv33_RELU_BN_Residuals_Add(Enc_3_MP, 192, range3d=range3d, name = "Encoder4")
    Enc_4_MP = UpSampling3D(size=(1, 2, 2))(Enc_4)
   
    Dec_1 = Conv33_RELU_BN_Residuals_Add(concatenate([Enc_3,Enc_4_MP]), 96, range3d=range3d, name = "Decoder1")
    Dec_1_US = UpSampling3D(size=(1, 2, 2))(Dec_1)

    Dec_2 = Conv33_RELU_BN_Residuals_Add(concatenate([Enc_2,Dec_1_US]), 48, range3d=range3d, name = "Decoder2")
    Dec_2_US = UpSampling3D(size=(1, 2, 2))(Dec_2)

    Dec_3 = Conv33_RELU_BN_Residuals_Add(concatenate([Enc_1,Dec_2_US]), 24, range3d=range3d, name = "Decoder3")

    OneOne_Conv = Conv3D(1, (1,1,1), activation=None)(Dec_3)
    
    output = Add()([pre_image_layer, OneOne_Conv])
    
    if not output3d:
        output = Conv3D(1, (1+2*range3d,1,1), activation=None)(output)

        
    model = Model(inputs = inputs, outputs = output)
        
    return model


from tensorflow import keras

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet(input_size = (256,256,1)):
    
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input(input_size)
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="linear")(u4)
    model = keras.models.Model(inputs, outputs)
    
    return model



def custom_VGG16(input_shape = (256,256,1), red_factor = 4, rescale=None):
    
    model = Sequential()
        
    model.add(tf.keras.layers.Input(input_shape))
    
    if rescale is not None:
        model.add(tf.keras.layers.Lambda(lambda x: x*(rescale)))        
    
    model.add(Conv2D(filters=int(64/red_factor),kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=int(64/red_factor),kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=int(128/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(128/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=int(256/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(256/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(256/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=int(512/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(512/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(512/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=int(512/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(512/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=int(512/red_factor), kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(units=64,activation="relu"))
    model.add(Dense(units=16,activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
  
    return model

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def discriminator_pix2pix(inputShape=[256, 256, 2], targetShape=[256, 256, 1]):
    
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=inputShape, name='input_image')
    tar = tf.keras.layers.Input(shape=targetShape, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)