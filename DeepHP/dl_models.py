#============================================================
#
#  Deep HistoPathology (DeepHP)
#  DL Models
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Activation
from keras.models import Model


def Inception_Depthwise_mod_v0(x, layers):
    # Inception module
    # https://arxiv.org/pdf/1409.4842.pdf
    #
    # Depthwise Separable convolutions
    # https://arxiv.org/pdf/1704.04861.pdf
    # Reduces the computational and memory footprint

    layer_input_size = int(x.shape[-1]) #TODO: use keras.backend.shape(x)

    # DW 5x5
    # Conv kernel size = 5x5
    conv_5x5 = Conv2D(layer_input_size, kernel_size=(5, 5),
                       activation='linear',
                       strides=1,
                       padding='same')(x)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)
    # Conv kernel size = 1x1
    conv_5x5_dw = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(conv_5x5)
    conv_5x5_dw = BatchNormalization()(conv_5x5_dw)
    conv_5x5_dw = Activation('relu')(conv_5x5_dw)

    # DW 3x3
    # Conv kernel size = 3x3
    conv_3x3 = Conv2D(layer_input_size, kernel_size=(3, 3),
                       activation='linear',
                       strides=1,
                       padding='same')(x)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)
    # Conv kernel size = 1x1
    conv_3x3_dw = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(conv_3x3)
    conv_3x3_dw = BatchNormalization()(conv_3x3_dw)
    conv_3x3_dw = Activation('relu')(conv_3x3_dw)

    # DW 1x1
    # Conv kernel size = 1x1
    conv_1x1 = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(x)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1_dw = Activation('relu')(conv_1x1)

    # DW MaxPool 3x3
    # MaxPool 3x3
    maxpool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # Conv kernel size = 1x1
    maxpool_3x3_dw = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(maxpool_3x3)
    maxpool_3x3_dw = BatchNormalization()(maxpool_3x3_dw)
    maxpool_3x3_dw = Activation('relu')(maxpool_3x3_dw)

    # Concatenation of all layer_input_size
    out = concatenate([conv_5x5_dw, conv_3x3_dw, conv_1x1_dw, maxpool_3x3_dw])

    return out

def Inception_Depthwise_mod_v1(x, layers):
    # Inception module
    # https://arxiv.org/pdf/1409.4842.pdf
    #
    # Depthwise Separable convolutions
    # https://arxiv.org/pdf/1704.04861.pdf
    # Reduces the computational and memory footprint

    layer_input_size = int(x.shape[-1]) #TODO: use keras.backend.shape(x)

    reduce_5x5_layers = int(layers / 3)
    reduce_3x3_layers = int(layers / 2)

    # DW 5x5
    # Conv kernel size = 5x5
    conv_5x5 = Conv2D(reduce_5x5_layers, kernel_size=(5, 5),
                       activation='linear',
                       strides=1,
                       padding='same')(x)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)
    # Conv kernel size = 1x1
    conv_5x5_dw = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(conv_5x5)
    conv_5x5_dw = BatchNormalization()(conv_5x5_dw)
    conv_5x5_dw = Activation('relu')(conv_5x5_dw)

    # DW 3x3
    # Conv kernel size = 3x3
    conv_3x3 = Conv2D(reduce_3x3_layers, kernel_size=(3, 3),
                       activation='linear',
                       strides=1,
                       padding='same')(x)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)
    # Conv kernel size = 1x1
    conv_3x3_dw = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(conv_3x3)
    conv_3x3_dw = BatchNormalization()(conv_3x3_dw)
    conv_3x3_dw = Activation('relu')(conv_3x3_dw)

    # DW 1x1
    # Conv kernel size = 1x1
    conv_1x1 = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(x)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1_dw = Activation('relu')(conv_1x1)

    # DW MaxPool 3x3
    # MaxPool 3x3
    maxpool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # Conv kernel size = 1x1
    maxpool_3x3_dw = Conv2D(layers, kernel_size=(1, 1),
                       activation='linear',
                       strides=1,
                       padding='same')(maxpool_3x3)
    maxpool_3x3_dw = BatchNormalization()(maxpool_3x3_dw)
    maxpool_3x3_dw = Activation('relu')(maxpool_3x3_dw)

    # Concatenation of all layer_input_size
    out = concatenate([conv_5x5_dw, conv_3x3_dw, conv_1x1_dw, maxpool_3x3_dw])

    return out

def Inception_v1_module(x, layers):
    # Inception module
    # https://arxiv.org/pdf/1409.4842.pdf

    reduce_5x5_layers = int(layers / 3)
    reduce_3x3_layers = int(layers / 2)

    # 5x5 Branch
    # Conv kernel size = 1x1
    conv_5x5_reduce = Conv2D(reduce_5x5_layers, kernel_size=(1, 1),
                             activation='linear',
                             strides=1,
                             padding='same')(x)
    conv_5x5_reduce = BatchNormalization()(conv_5x5_reduce)
    conv_5x5_reduce = Activation('relu')(conv_5x5_reduce)

    # Conv kernel size = 5x5
    conv_5x5 = Conv2D(layers, kernel_size=(5, 5),
                      activation='linear',
                      strides=1,
                      padding='same')(conv_5x5_reduce)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)

    # 3x3 Branch
    # Conv kernel size = 1x1
    conv_3x3_reduce = Conv2D(reduce_3x3_layers, kernel_size=(1, 1),
                             activation='linear',
                             strides=1,
                             padding='same')(x)
    conv_3x3_reduce = BatchNormalization()(conv_3x3_reduce)
    conv_3x3_reduce = Activation('relu')(conv_3x3_reduce)

    # Conv kernel size = 3x3
    conv_3x3 = Conv2D(layers, kernel_size=(3, 3),
                      activation='linear',
                      strides=1,
                      padding='same')(conv_3x3_reduce)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)

    # 1x1 Branch
    # Conv kernel size = 1x1
    conv_1x1 = Conv2D(layers, kernel_size=(1, 1),
                      activation='linear',
                      strides=1,
                      padding='same')(x)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)

    # MaxPool 3x3 Branch
    # MaxPool 3x3
    maxpool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # Conv kernel size = 1x1
    maxpool_3x3 = Conv2D(layers, kernel_size=(1, 1),
                            activation='linear',
                            strides=1,
                            padding='same')(maxpool_3x3)
    maxpool_3x3 = BatchNormalization()(maxpool_3x3)
    maxpool_3x3 = Activation('relu')(maxpool_3x3)

    # Concatenation of all layer_input_size
    out = concatenate([conv_5x5, conv_3x3, conv_1x1, maxpool_3x3])

    return out

def Inception_v1_module_non_bn(x, layers):
    # Inception module
    # https://arxiv.org/pdf/1409.4842.pdf

    reduce_5x5_layers = int(layers / 3)
    reduce_3x3_layers = int(layers / 2)

    # 5x5 Branch
    # Conv kernel size = 1x1
    conv_5x5_reduce = Conv2D(reduce_5x5_layers, kernel_size=(1, 1),
                             activation='linear',
                             strides=1,
                             padding='same')(x)
    #conv_5x5_reduce = BatchNormalization()(conv_5x5_reduce)
    conv_5x5_reduce = Activation('relu')(conv_5x5_reduce)

    # Conv kernel size = 5x5
    conv_5x5 = Conv2D(layers, kernel_size=(5, 5),
                      activation='linear',
                      strides=1,
                      padding='same')(conv_5x5_reduce)
    #conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)

    # 3x3 Branch
    # Conv kernel size = 1x1
    conv_3x3_reduce = Conv2D(reduce_3x3_layers, kernel_size=(1, 1),
                             activation='linear',
                             strides=1,
                             padding='same')(x)
    #conv_3x3_reduce = BatchNormalization()(conv_3x3_reduce)
    conv_3x3_reduce = Activation('relu')(conv_3x3_reduce)

    # Conv kernel size = 3x3
    conv_3x3 = Conv2D(layers, kernel_size=(3, 3),
                      activation='linear',
                      strides=1,
                      padding='same')(conv_3x3_reduce)
    #conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)

    # 1x1 Branch
    # Conv kernel size = 1x1
    conv_1x1 = Conv2D(layers, kernel_size=(1, 1),
                      activation='linear',
                      strides=1,
                      padding='same')(x)
    #conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)

    # MaxPool 3x3 Branch
    # MaxPool 3x3
    maxpool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # Conv kernel size = 1x1
    maxpool_3x3 = Conv2D(layers, kernel_size=(1, 1),
                            activation='linear',
                            strides=1,
                            padding='same')(maxpool_3x3)
    #maxpool_3x3 = BatchNormalization()(maxpool_3x3)
    maxpool_3x3 = Activation('relu')(maxpool_3x3)

    # Concatenation of all layer_input_size
    out = concatenate([conv_5x5, conv_3x3, conv_1x1, maxpool_3x3])

    return out

def Inception_v1_1_module(x, layers):
    # Inception module
    # https://arxiv.org/pdf/1409.4842.pdf

    layer_input_size = int(x.shape[-1])  # TODO: use keras.backend.shape(x)

    reduce_5x5_layers = layer_input_size
    reduce_3x3_layers = layer_input_size

    # reduce_5x5_layers = int(layers / 3)
    # reduce_3x3_layers = int(layers / 2)

    # 5x5 Branch
    # Conv kernel size = 1x1
    conv_5x5_reduce = Conv2D(reduce_5x5_layers, kernel_size=(1, 1),
                             activation='linear',
                             strides=1,
                             padding='same')(x)
    conv_5x5_reduce = BatchNormalization()(conv_5x5_reduce)
    conv_5x5_reduce = Activation('relu')(conv_5x5_reduce)

    # Conv kernel size = 5x5
    conv_5x5 = Conv2D(layers, kernel_size=(5, 5),
                      activation='linear',
                      strides=1,
                      padding='same')(conv_5x5_reduce)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)

    # 3x3 Branch
    # Conv kernel size = 1x1
    conv_3x3_reduce = Conv2D(reduce_3x3_layers, kernel_size=(1, 1),
                             activation='linear',
                             strides=1,
                             padding='same')(x)
    conv_3x3_reduce = BatchNormalization()(conv_3x3_reduce)
    conv_3x3_reduce = Activation('relu')(conv_3x3_reduce)

    # Conv kernel size = 3x3
    conv_3x3 = Conv2D(layers, kernel_size=(3, 3),
                      activation='linear',
                      strides=1,
                      padding='same')(conv_3x3_reduce)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)

    # 1x1 Branch
    # Conv kernel size = 1x1
    conv_1x1 = Conv2D(layers, kernel_size=(1, 1),
                      activation='linear',
                      strides=1,
                      padding='same')(x)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)

    # MaxPool 3x3 Branch
    # MaxPool 3x3
    maxpool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # Conv kernel size = 1x1
    maxpool_3x3 = Conv2D(layers, kernel_size=(1, 1),
                            activation='linear',
                            strides=1,
                            padding='same')(maxpool_3x3)
    maxpool_3x3 = BatchNormalization()(maxpool_3x3)
    maxpool_3x3 = Activation('relu')(maxpool_3x3)

    # Concatenation of all layer_input_size
    out = concatenate([conv_5x5, conv_3x3, conv_1x1, maxpool_3x3])

    return out

def deepHP_model_v0():
    num_clases = 2
    input_shape = (256, 256, 1) # I will assuame a 256x256x1 input shape
    input_img = Input(shape=input_shape)

    tensors = Inception_Depthwise_mod_v0(input_img, layers=16) #features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #128x128
    tensors = Inception_Depthwise_mod_v0(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #64x64
    tensors = Inception_Depthwise_mod_v0(tensors, layers=128) #features number 512
    tensors = Inception_Depthwise_mod_v0(tensors, layers=128) #features number 512
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #32x32
    tensors = Inception_Depthwise_mod_v0(tensors, layers=256) #features number 1024
    tensors = Inception_Depthwise_mod_v0(tensors, layers=256) #features number 1024
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #16x16
    tensors = Inception_Depthwise_mod_v0(tensors, layers=512) #features number 2048
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #8x8
    tensors = Inception_Depthwise_mod_v0(tensors, layers=512) #features number 2048
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(512, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(512, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_v1():
    num_clases = 2
    input_shape = (256, 256, 1) # I will assuame a 256x256x1 input shape
    input_img = Input(shape=input_shape)

    tensors = Inception_Depthwise_mod_v0(input_img, layers=16) #features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #128x128
    tensors = Inception_Depthwise_mod_v0(tensors, layers=32) #features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #64x64
    tensors = Inception_Depthwise_mod_v0(tensors, layers=64) #features number 256
    tensors = Inception_Depthwise_mod_v0(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #32x32
    tensors = Inception_Depthwise_mod_v0(tensors, layers=128) #features number 512
    tensors = Inception_Depthwise_mod_v0(tensors, layers=128) #features number 512
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #16x16
    tensors = Inception_Depthwise_mod_v0(tensors, layers=256) #features number 1024
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #8x8
    tensors = Inception_Depthwise_mod_v0(tensors, layers=256) #features number 1024
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_BC_V0_DW_V0(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_Depthwise_mod_v0(input_img, layers=16) #features number 64
    tensors = Inception_Depthwise_mod_v0(tensors, layers=16)  # features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_Depthwise_mod_v0(tensors, layers=32) #features number 128
    tensors = Inception_Depthwise_mod_v0(tensors, layers=32)  # features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_Depthwise_mod_v0(tensors, layers=64) #features number 256
    tensors = Inception_Depthwise_mod_v0(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_Depthwise_mod_v0(tensors, layers=128) #features number 512
    tensors = Inception_Depthwise_mod_v0(tensors, layers=128) #features number 512
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_BC_V0_DW_V1(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_Depthwise_mod_v1(input_img, layers=16) #features number 64
    tensors = Inception_Depthwise_mod_v1(tensors, layers=16)  # features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_Depthwise_mod_v1(tensors, layers=32) #features number 128
    tensors = Inception_Depthwise_mod_v1(tensors, layers=32)  # features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_Depthwise_mod_v1(tensors, layers=64) #features number 256
    tensors = Inception_Depthwise_mod_v1(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_Depthwise_mod_v1(tensors, layers=128) #features number 512
    tensors = Inception_Depthwise_mod_v1(tensors, layers=128) #features number 512
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs=[input_img], outputs=predictions)
    return model

def deepHP_model_BC_V0_inception(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    conv_dp = 0.05

    tensors = Inception_v1_module(input_img, layers=16) #features number 64
    #tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=16)  # features number 64
    #tensors = Dropout(conv_dp)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_v1_module(tensors, layers=32) #features number 128
    #tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=32)  # features number 128
    #tensors = Dropout(conv_dp)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_module(tensors, layers=64) #features number 256
    #tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=64) #features number 256
    #tensors = Dropout(conv_dp)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_module(tensors, layers=128) #features number 512
    #tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=128) #features number 512
    #tensors = Dropout(conv_dp)(tensors)
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear',
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=regularizers.l2(1e-4),
                    # activity_regularizer=regularizers.l2(1e-5)
                    )(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear',
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=regularizers.l2(1e-4),
                    # activity_regularizer=regularizers.l2(1e-5)
                    )(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs=[input_img], outputs=predictions)
    return model


def deepHP_model_BC_V0_1_inception(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    conv_dp = 0.05

    tensors = Inception_v1_module(input_img, layers=64) #features number 64
    # tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=64)  # features number 64
    # tensors = Dropout(conv_dp)(tensors)
    tensors = MaxPooling2D(pool_size=(2, 2))(tensors) #25x125
    tensors = Inception_v1_module(tensors, layers=128) #features number 128
    # tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=128)  # features number 128
    # tensors = Dropout(conv_dp)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_module(tensors, layers=256) #features number 256
    # tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=256) #features number 256
    # tensors = Dropout(conv_dp)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_module(tensors, layers=512) #features number 512
    # tensors = Dropout(conv_dp)(tensors)
    tensors = Inception_v1_module(tensors, layers=512) #features number 512
    # tensors = Dropout(conv_dp)(tensors)
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear',
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=regularizers.l2(1e-4),
                    # activity_regularizer=regularizers.l2(1e-5)
                    )(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear',
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=regularizers.l2(1e-4),
                    # activity_regularizer=regularizers.l2(1e-5)
                    )(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs=[input_img], outputs=predictions)
    return model


def deepHP_model_BC_V0_inception_non_bn(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_v1_module_non_bn(input_img, layers=16) #features number 64
    tensors = Inception_v1_module_non_bn(tensors, layers=16)  # features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_v1_module_non_bn(tensors, layers=32) #features number 128
    tensors = Inception_v1_module_non_bn(tensors, layers=32)  # features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_module_non_bn(tensors, layers=64) #features number 256
    tensors = Inception_v1_module_non_bn(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_module_non_bn(tensors, layers=128) #features number 512
    tensors = Inception_v1_module_non_bn(tensors, layers=128) #features number 512
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    #tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    #tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_BC_V0_inception_NO_FC(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_v1_module(input_img, layers=16) #features number 64
    tensors = Inception_v1_module(tensors, layers=16)  # features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_v1_module(tensors, layers=32) #features number 128
    tensors = Inception_v1_module(tensors, layers=32)  # features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_module(tensors, layers=64) #features number 256
    tensors = Inception_v1_module(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_module(tensors, layers=128) #features number 512
    tensors = Inception_v1_module(tensors, layers=128) #features number 512
    # FC1 substitute
    tensors = Conv2D(filters=256, kernel_size=(6,6), activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2 substitute
    tensors = Conv2D(filters=256, kernel_size=(1, 1), activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    #predictions = Conv2D(filters=num_clases, kernel_size=(1,1), activation='softmax')(tensors)
    tensors = Flatten()(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_BC_V1_inception(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_v1_module(input_img, layers=32) #features number 64
    tensors = Inception_v1_module(tensors, layers=32)  # features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_v1_module(tensors, layers=64) #features number 128
    tensors = Inception_v1_module(tensors, layers=64)  # features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_module(tensors, layers=128) #features number 256
    tensors = Inception_v1_module(tensors, layers=128) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_module(tensors, layers=256) #features number 512
    tensors = Inception_v1_module(tensors, layers=256) #features number 512
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_BC_v1_DP_inception(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_v1_module(input_img, layers=32) #features number 64
    tensors = Dropout(0.2)(tensors)
    tensors = Inception_v1_module(tensors, layers=32)  # features number 64
    tensors = Dropout(0.2)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_v1_module(tensors, layers=64) #features number 128
    tensors = Dropout(0.2)(tensors)
    tensors = Inception_v1_module(tensors, layers=64)  # features number 128
    tensors = Dropout(0.2)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_module(tensors, layers=128) #features number 256
    tensors = Dropout(0.2)(tensors)
    tensors = Inception_v1_module(tensors, layers=128) #features number 256
    tensors = Dropout(0.2)(tensors)
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_module(tensors, layers=256) #features number 512
    tensors = Dropout(0.2)(tensors)
    tensors = Inception_v1_module(tensors, layers=256) #features number 512
    tensors = Dropout(0.2)(tensors)
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model

def deepHP_model_BC_inception_v1_1(): # BC = version
    num_clases = 2
    input_shape = (50, 50, 3)
    input_img = Input(shape=input_shape)

    tensors = Inception_v1_1_module(input_img, layers=16) #features number 64
    tensors = Inception_v1_1_module(tensors, layers=16)  # features number 64
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #25x125
    tensors = Inception_v1_1_module(tensors, layers=32) #features number 128
    tensors = Inception_v1_1_module(tensors, layers=32)  # features number 128
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #12x12
    tensors = Inception_v1_1_module(tensors, layers=64) #features number 256
    tensors = Inception_v1_1_module(tensors, layers=64) #features number 256
    tensors = MaxPooling2D(pool_size=(2,2))(tensors) #6x6
    tensors = Inception_v1_1_module(tensors, layers=128) #features number 512
    tensors = Inception_v1_1_module(tensors, layers=128) #features number 512
    # FC1
    tensors = Flatten()(tensors)
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    # FC2
    tensors = Dense(256, activation='linear')(tensors)
    tensors = BatchNormalization()(tensors)
    tensors = Dropout(0.4)(tensors)
    tensors = Activation('relu')(tensors)
    predictions = Dense(num_clases, activation='softmax')(tensors)
    model = Model(inputs= [input_img], outputs=predictions)
    return model


# # ==================
# # LOAD THE DL MODEL
# # ==================
# model = deepHP_model_v1()
# model.summary()
# gbytes = Utils.get_model_memory_usage(batch_size=1, model=model)
# print('Aproximated amount of GPU RAM required: ' + str(gbytes) + ' GB')
