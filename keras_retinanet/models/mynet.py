"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings

import keras,keras_retinanet
from ..models import retinanet
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K

custom_objects = retinanet.custom_objects.copy()

allowed_backbones = ['mynet']

def validate_backbone(backbone):
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

def droneNet(inputs=None, include_top=True, classes=10, *args, **kwargs):
    if inputs is None :
        if K.image_data_format() == 'channels_first':
            input_shape = Input(shape=(3, 224, 224))
        else:
            input_shape = Input(shape=(224, 224, 3))
    else:
        input_shape=inputs

    outputs = []

    x = Conv2D(32, (3, 3), strides=(1, 1),use_bias=False)(input_shape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    outputs.append(x)

    for i in range(3):
        x = Conv2D(64*(2**i), (3, 3), strides=(1, 1),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        outputs.append(x)

    x = Conv2D(256, (3, 3), strides=(1, 1),use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    outputs.append(x)
    

    if include_top:
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='sigmoid')(x)
        return Model(inputs=input_shape, outputs=x, *args, **kwargs)
    else:
        return Model(inputs=input_shape, outputs=outputs, *args, **kwargs)


def mynet_retinanet(num_classes, backbone=',mynet', inputs=None, modifier=None, **kwargs):

    # choose default input
    if inputs is None:
        inputs = Input(shape=(None, None, 3))
        
    mynet=droneNet(inputs=inputs,include_top=False)

    # invoke modifier if given
    if modifier:
        mynet = modifier(mynet)

    # create the full model
    return retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_layers=mynet.outputs[2:], **kwargs)