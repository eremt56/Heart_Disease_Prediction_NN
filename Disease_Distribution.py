import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math


dataSetUp=pd.read_csv("heart-data.csv")

data = np.array(dataSetUp)

graphData = data[:, 11]

np.random.shuffle(graphData)

plt.plot(graphData)
plt.savefig("DiseaseDistribution")