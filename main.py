from sklearn.model_selection import train_test_split
import os
from dataset_load import *
from models import *
import glob



AllDatasetFolder ='Dataset/' # use your path

 
epoch = 30
batch_size = 16

print("..................................")
print("......Start training on  : dataset.............")
print("................................................Please wait a while...........................")
print("..................................")

TrainData = AllDatasetFolder

x_train, x_test, y_train, y_test  = loadDataset(TrainData)


Models = [proposed_model]

numberofmodels=len(Models)+1

for i in Models:
    i(x_train, x_test, y_train, y_test, numberofmodels, epoch, batch_size)