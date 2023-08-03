from dataset_load import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import array
import pandas as pd
import numpy as np
from dataset_load import *
import csv
import sklearn.metrics as metrics
import matplotlib.pyplot as ptl
from sklearn.svm import SVR
from math import sqrt
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

C = 0
def losses_function(y_test, y_pred, model_name, numberofmodels, model, x_test):
    
    global C
    
    
    testing_accurarcy_loss = model.evaluate(x_test,  y_test)
	
    precision_score_micro = precision_score(y_test, y_pred, average='micro')
    recall_score_micro = recall_score(y_test, y_pred, average='micro')
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    
    precision_score_macro = precision_score(y_test, y_pred, average='macro')
    recall_score_macro = recall_score(y_test, y_pred, average='macro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    
    precision_score_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_score_weighted =  recall_score(y_test, y_pred, average='weighted')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')
    
    precision_score_micro = round(precision_score_micro, 4)
    recall_score_micro = round(recall_score_micro, 4)
    f1_score_micro = round(f1_score_micro, 4)
    precision_score_macro = round(precision_score_macro, 4)
    
    recall_score_macro = np.round(recall_score_macro, 4)
    f1_score_macro = np.round(f1_score_macro, 4)
    precision_score_weighted = np.round(precision_score_weighted, 4)
    recall_score_weighted = np.round(recall_score_weighted, 4)
    f1_score_weighted = np.round(f1_score_weighted, 4)
    
    print (testing_accurarcy_loss)
    
    # field names
    header = ["Model_Name", "precision_score_micro",  "recall_score_micro" , "f1_score_micro", "precision_score_macro", "recall_score_macro", "f1_score_macro","precision_score_weighted", "recall_score_weighted", "f1_score_weighted", "testing_loss", "testing accuracy"]
     
    # data rows of csv file
    data = [model_name, precision_score_micro, recall_score_micro, f1_score_micro, precision_score_macro, recall_score_macro, f1_score_macro, precision_score_weighted, recall_score_weighted, f1_score_weighted, testing_accurarcy_loss[0],testing_accurarcy_loss[1]]
     
     
    # writing to csv file
    with open("Results/result_file.csv" , 'a', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
		
        if C%numberofmodels==0:
            
            writer.writerow(header)
			
            C+=1
            
        writer.writerow(data)
    C+=1
    
    return (y_test, y_pred, model_name)