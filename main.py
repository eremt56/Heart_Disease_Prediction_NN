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

dataSetUp = pd.get_dummies(dataSetUp)

data=np.array(dataSetUp)

# np.random.shuffle(data)

data_set=data.T

for i in range(data_set.shape[0]):
    data_set[i, :] = model.normalize(data_set[i, :])

val_set_answer = data_set[20:21, 0:100]

val_set = data_set[0:21, 0:100]

training_set = data_set[0:21, 100:917]

training_set_answer = data_set[20:21, 100:917]

model.__init__()

model.training(training_set.T, training_set_answer.T, val_set.T, val_set_answer.T)





