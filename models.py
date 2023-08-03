from tensorflow.keras.layers import Input
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from dataset_load import *
from losses import *
from plot import *
import numpy as np
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as ptl
from tensorflow.keras.models import Model
import datetime
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
    
inp = 128

def proposed_model(x_train, x_test, y_train, y_test, numberofmodels, epoch, batch_size):
    
    model_name= "ConvNeXtTiny"
    
    conv_base = tf.keras.applications.ConvNeXtTiny( include_top = False, input_shape= (inp,inp,3))
    
    num_layers_to_remove = 1
    for _ in range(num_layers_to_remove):
        conv_base.layers.pop()
        
    new_model = Model(inputs=conv_base.inputs, outputs=conv_base.layers[-65].output)
        
    model = models.Sequential()
    
    model.add(new_model)
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(128))

    model.add(Dense(2,activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    model.summary()
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test)) 
    
    
    model.save("new_One.h5")
    
    y_pred = model.predict(x_test, verbose=1, batch_size=batch_size)
    
    y_pred=np.argmax(y_pred, axis=1)
    
    my_plot(history, model_name)
    
    losses_function(y_test, y_pred, model_name, numberofmodels, model, x_test)

