import numpy as np
import random as rn
import scipy.io as sio
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import VarianceScaling
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras.activations import *
from keras.models import Model
from keras.regularizers import L1L2

def SE(input,D,units=32):
    x = input
    #squeeze
    if D == 2:
        output = GlobalAveragePooling2D()(input)
    else:
        output = GlobalAveragePooling1D()(input)
    
    #excitation
    output = Dense(units//4)(output)
    output = ReLU()(output)
    output = Dense(units)(output)
    output = Activation('sigmoid')(output)
    
    if D==2:
        output = Reshape((1, 1, units))(output)
    else:
        output = Reshape((1, units))(output)
    #scale
    output = multiply([x, output])

    return output

def conv1d(filters,kernel_size,input_layer):
    x=Convolution1D(filters=filters,kernel_size=kernel_size,padding='same',kernel_regularizer=None)(input_layer)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.3)(x)
    
    return x

def conv2d(filters,kernel_size,input_layer):
    x=Convolution2D(filters=filters,kernel_size=kernel_size,padding='same',kernel_regularizer=None)(input_layer)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.3)(x)
    
    return x

def ResBlock2d(filters,input_layer):
    x=input_layer
    output=conv2d(filters,(17,1),input_layer)
    output=conv2d(filters,(11,1),output)
    output=conv2d(filters,(5,1),output)
    output = SE(output,2,filters)

    x=conv2d(filters,(1,1),x)

    output=output+x

    output=LeakyReLU(alpha=0.3)(output)

    return output

def ResBlock1d(filters,input_layer):
    x=input_layer
    output=conv1d(filters,17,input_layer)
    output=conv1d(filters,11,output)
    output=conv1d(filters,5,output)
    output = SE(output,1,filters)

    x=conv1d(filters,1,x)
    output = output+x

    output=LeakyReLU(alpha=0.3)(output)
    output=AveragePooling1D(3)(output)

    return output

def bolck(filters,input_layer):
    x=ResBlock2d(filters,input_layer)
    x=Reshape((int(input_layer.shape[1]/2),filters))(input_layer)
    x=ResBlock1d(filters,x)
    x=ResBlock1d(2*filters,x)
    x=ResBlock1d(4*filters,x)

    x=GlobalAveragePooling1D()(x)
    return x

def mtmlsenet(input_shape,ptbxlptb):

    input_layer1= Input(shape=input_shape)
    input_layer2= Input(shape=input_shape)
    input_layer3= Input(shape=input_shape)
    input_layer4= Input(shape=input_shape)
    input_layer5= Input(shape=input_shape)   
    input_layer6= Input(shape=input_shape)   
    input_layer7= Input(shape=input_shape)   
    input_layer8= Input(shape=input_shape)   
    input_layer9= Input(shape=input_shape)   
    input_layer10= Input(shape=input_shape)   
    input_layer11= Input(shape=input_shape)   
    input_layer12= Input(shape=input_shape)

    input1=tf.expand_dims(input_layer1,axis=-1)
    input2=tf.expand_dims(input_layer2,axis=-1)
    input3=tf.expand_dims(input_layer3,axis=-1)
    input4=tf.expand_dims(input_layer4,axis=-1)
    input5=tf.expand_dims(input_layer5,axis=-1)
    input6=tf.expand_dims(input_layer6,axis=-1)
    input7=tf.expand_dims(input_layer7,axis=-1)
    input8=tf.expand_dims(input_layer8,axis=-1)
    input9=tf.expand_dims(input_layer9,axis=-1)
    input10=tf.expand_dims(input_layer10,axis=-1)
    input11=tf.expand_dims(input_layer11,axis=-1)
    input12=tf.expand_dims(input_layer12,axis=-1)

    x1 = conv2d(2,(20,1),input1)
    x2 = conv2d(2,(20,1),input2)
    x3 = conv2d(2,(20,1),input3)
    x4 = conv2d(2,(20,1),input4)
    x5 = conv2d(2,(20,1),input5)
    x6 = conv2d(2,(20,1),input6)
    x7 = conv2d(2,(20,1),input7)
    x8 = conv2d(2,(20,1),input8)
    x9 = conv2d(2,(20,1),input9)
    x10 = conv2d(2,(20,1),input10)
    x11 = conv2d(2,(20,1),input11)
    x12 = conv2d(2,(20,1),input12)

    x1 = bolck(4,x1)
    x2 = bolck(4,x2)
    x3 = bolck(4,x3)
    x4 = bolck(4,x4)
    x5 = bolck(4,x5)
    x6 = bolck(4,x6)
    x7 = bolck(4,x7)
    x8 = bolck(4,x8)
    x9 = bolck(4,x9)
    x10 = bolck(4,x10)
    x11 = bolck(4,x11)
    x12 = bolck(4,x12)

    x=concatenate([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12])
    x=Dropout(0.5)(x)
    x=Flatten()(x)

    if(int(ptbxlptb)==0):
        detection_output = Dense(2, activation='sigmoid',kernel_regularizer=L1L2(l1=0.0,l2=0.001),name='detection')(x)
        location_5_output = Dense(5, activation='softmax',kernel_regularizer=L1L2(l1=0.0,l2=0.001),name='location_5')(x)
        location_7_output = Dense(7, activation='softmax',kernel_regularizer=L1L2(l1=0.0,l2=0.001),name='location_7')(x)
    else:
        detection_output = Dense(2, activation='sigmoid',kernel_regularizer=L1L2(l1=0.0,l2=0.001),name='detection')(x)
        location_5_output = Dense(5, activation='sigmoid',kernel_regularizer=L1L2(l1=0.0,l2=0.001),name='location_5')(x)
        location_7_output = Dense(7, activation='sigmoid',kernel_regularizer=L1L2(l1=0.0,l2=0.001),name='location_7')(x)


    model = keras.Model(inputs=[input_layer1,input_layer2,input_layer3,input_layer4,input_layer5,input_layer6,input_layer7,input_layer8,input_layer9,input_layer10,input_layer11,input_layer12], outputs=[detection_output,location_5_output,location_7_output])
    return model
