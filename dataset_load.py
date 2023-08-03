from losses import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from numpy import array
import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as ptl
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def loadDataset(path):
    
    folders = glob.glob(path+"train/*")

    img_list = []
    label_list=[]
    
    for folder in folders:
        print(folder)
        for img in glob.glob(folder+r"/*.jpg"):
            n= cv2.imread(img)
            class_num = folders.index(folder)
            label_list.append(class_num)
            resized = cv2.resize(n, (128, 128))
            img_list.append(resized)
        
    #Splitting the data based on scikit-learn Library
    x_train, x_test, y_train, y_test = train_test_split(img_list, label_list, test_size=0.2, random_state=1) 
   
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
       
    
    #View the data
    print ("training_set", x_train.shape)
    print ("training_set", y_train.shape)
    print ("validation_set",x_test.shape)
    print ("validation_set",y_test.shape)
    

    return  x_train, x_test, y_train, y_test 
























    