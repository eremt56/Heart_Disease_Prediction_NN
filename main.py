#import libraries
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import Model as Model

model = Model.Model()


#Process and Standardize Raw Data

dataSetUp=pd.read_csv("heart-data.csv")

data=np.array(dataSetUp)

m, n = data.T.shape

data_set=data.T

data_set[0] = model.normalize(data_set[0])

data_set[1] = np.where(data_set[1] == 'M', 1, data_set[1])
data_set[1] = np.where(data_set[1] == 'F', -1, data_set[1])

data_set[2] = np.where(data_set[2]=="ATA", 1, data_set[2])
data_set[2] = np.where(data_set[2]=="ASY", -1, data_set[2])
data_set[2] = np.where(data_set[2]=="NAP", 2, data_set[2])
data_set[2] = np.where(data_set[2]=="TA", -2, data_set[2])

data_set[2] = model.normalize(data_set[2])

data_set[3] = model.normalize(data_set[3])

data_set[4] = model.normalize(data_set[4])

data_set[5] = model.normalize(data_set[5])

data_set[6] = np.where(data_set[6] == 'Normal', -1, data_set[6])
data_set[6] = np.where(data_set[6] == "ST", 0, data_set[6])
data_set[6] = np.where(data_set[6] == "LVH", 1, data_set[6])

data_set[6] = model.normalize(data_set[6])

data_set[7] = model.normalize(data_set[7])

data_set[8] = np.where(data_set[8] == 'Y', 1, data_set[8])
data_set[8] = np.where(data_set[8] == 'N', -1, data_set[8])

data_set[9] = model.normalize(data_set[9])

data_set[10] = np.where(data_set[10] == "Up", 1, data_set[10])
data_set[10] = np.where(data_set[10] == "Flat", 0, data_set[10])
data_set[10] = np.where(data_set[10] == "Down", -1, data_set[10])

val_set_answer = data_set[11:12, 0:100]

val_set = data_set[0:11, 0:100]

training_set = data_set[0:11, 100:917]

training_set_answer = data_set[11:12, 100:917]

model.__init__()

model.training(training_set.T, training_set_answer.T, val_set.T, val_set_answer.T)


