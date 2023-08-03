from dataset_load import *
from losses import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset_load import *
import matplotlib.pyplot as ptl

def my_plot(history, model_name):
    
    f, ax = plt.subplots()
    ax.plot([None] + history.history['accuracy'], 'o-')
    ax.plot([None] + history.history['val_accuracy'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train acc', 'Val acc'], loc = 0)
    ax.set_title('Training/Validation acc per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.savefig("Results/"+ model_name+"_Accuracy.JPG")
    f, ax = plt.subplots()
    ax.plot([None] + history.history['loss'], 'o-')
    ax.plot([None] + history.history['val_loss'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train loss', "Val loss"], loc = 1)
    ax.set_title('Training/Validation Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig("Results/"+ model_name +"_Loss.JPG")
    plt.show() 
