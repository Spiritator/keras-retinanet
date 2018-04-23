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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import metrics

custom_objects = retinanet.custom_objects.copy()

allowed_backbones = ['mynet']

def validate_backbone(backbone):
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

def initialize_mynet(input_shape = (None, None, 3), output_shape=4):
    mynet = Sequential()
    
    mynet.add(Conv2D(32, (3, 3), input_shape=input_shape))
    mynet.add(BatchNormalization())
    mynet.add(Activation('relu'))
    mynet.add(MaxPooling2D(pool_size=(2, 2)))
    
    mynet.add(Conv2D(64, (3, 3)))
    mynet.add(BatchNormalization())
    mynet.add(Activation('relu'))
    mynet.add(MaxPooling2D(pool_size=(2, 2)))
    
    mynet.add(Conv2D(128, (3, 3)))
    mynet.add(BatchNormalization())
    mynet.add(Activation('relu'))
    mynet.add(MaxPooling2D(pool_size=(2, 2)))
    
    mynet.add(Conv2D(256, (3, 3)))
    mynet.add(BatchNormalization())
    mynet.add(Activation('relu'))
    mynet.add(MaxPooling2D(pool_size=(2, 2)))
    
    mynet.add(Conv2D(256, (3, 3)))
    mynet.add(BatchNormalization())
    mynet.add(Activation('relu'))
    mynet.add(MaxPooling2D(pool_size=(2, 2)))
    mynet.add(Dropout(0.5))
    
    return mynet


def mynet_retinanet(num_classes, backbone=',mynet', inputs=None, modifier=None, **kwargs):

    # choose default input
    if inputs is None:
        inputs = (None, None, 3)
        
    mynet=initialize_mynet(input_shape=inputs)

    # invoke modifier if given
    if modifier:
        mynet = modifier(mynet)

    # create the full model
    return retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_layers=mynet.outputs, **kwargs)