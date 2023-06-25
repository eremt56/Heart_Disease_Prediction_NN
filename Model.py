#import libraries
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

# Algorithms:
class Model:

    
    weight1 = np.array([])
    bias1 = np.array([])

    weight2 = np.array([])
    bias2 = np.array([])

    output = np.array([])
    outputBias = np.array([])

    tempWeight1 = np.array([])
    tempBias1 = np.array([])

    tempWeight2 = np.array([])
    tempBias2 = np.array([])

    tempOutput = np.array([])
    tempOutputBias = np.array([])



    #initializing weights and biases:
    @staticmethod
    def __init__():
        '''Initialize all weights and biases:
        2 Hidden Layer
        1 Output Layer'''
        
        weight1 = (np.random.rand(11, 5) * 2) - 1
        bias1 = (np.random.randn(5,1)*2)-1

        weight2 = (np.random.rand(5, 1) * 2) - 1
        bias2 = (np.random.randn(1,1)*2)-1

        output = (np.random.rand(1, 1) * 2) - 1
        outputBias = (np.random.rand(1, 1) * 2) - 1

        tempWeight1 = weight1
        tempBias1 = bias1

        tempWeight2 = weight2
        tempBias2 = bias2

        tempOutput = output
        tempOutputBias = outputBias

        return weight1, bias1, weight2, bias2, output, outputBias
    
    def calculate(dataPoint):
        Z1 = weight1.dot(dataPoint) + bias1
        A1 = map(sigmoid, Z1)

        Z2 = weight2.T.dot(A1) + bias2
        A2 = map(sigmoid, Z2)

        Z3 = output.T.dot(A2) + outputBias
        A3 = map(sigmoid, Z3)

        return A3

        
    # BackPropogation Algorithm Begins:

    # Forward Prop:
    def forwardProp(trainingData):
        Z1 = weight1.dot(trainingData) + bias1
        A1 = map(sigmoid, Z1)

        Z2 = weight2.T.dot(A1) + bias2
        A2 = map(sigmoid, Z2)

        Z3 = output.T.dot(A2) + outputBias
        A3 = map(sigmoid, Z3)

        return Z1, A1, Z2, A2, Z3, A3
    
    def eval(val_set, val_set_answer):
        
        transfer = 0

        for i in val_set:
            Z1 = weight1.T.dot(i) + bias1
            A1 = map(sigmoid, Z1)

            Z2 = weight2.T.dot(A1) + bias2
            A2 = map(sigmoid, Z2)

            Z3 = output.T.dot(A2) + outputBias
            A3 = map(sigmoid, Z3)

            transfer += lossFunction(A3, i)

        return transfer/val_set[0].size


    # Back Prop:

    def backProp(Z1, A1, Z2, A2, Z3, A3, answer, input):
        
        val = -1
        
        third2 = deriv_lossFunction(answer, A3)
        third1 = deriv_sigmoid(Z3)
    
        
        temp3 = A2.dot(third2 * third1)

        temp3Bias = third2 * third1

        temp2 = A1.dot(((third2 * third1).dot(output)).dot(A2.dot((A2 * val) + 1)))
    
        temp2Bias = ((third2 * third1).dot(output)).dot(A2.dot((A2 * val) + 1))

        temp1 = input.dot((weight2.dot(((third2 * third1).dot(output)).dot(A2.dot((A2*-1) + 1)))).dot(A1.dot((A1*-1) + 1)))
    
        temp1Bias = (weight2.dot(((third2 * third1).dot(output)).dot(A2.dot((A2*-1) + 1)))).dot(A1.dot((A1*-1) + 1))
    

        tempOutput = tempOutput - 0.01 * temp3
    
        tempOutputBias = tempOuputBias - 0.01 * temp3Bias
    
        tempWeight2 = tempWeight2 - 0.01 * temp2
    
        tempBias2 = tempBias2 - 0.01 * temp2Bias

        tempWeight1 = tempWeight1 - 0.01 * temp1

        tempBias1 = tempBias1 - 0.01 * temp1Bias

    @staticmethod
    def training(trainingData, trainingAnswer,  val_set, val_set_answer):

        epoch = 0
        count = 0
        loss = float()

        for x in trainingData:
            
            if count > 27:

                weight1 = tempWeight1
                    
                weight2= tempWeight2

                output = tempOutput

                outputBias = tempOutputBias

                bias1 = tempBias1

                bias2 = tempBias2

                outputBias = tempOutputBias

                loss = eval(val_set, val_set_answer)

                epoch += 1

                print("Loss on epoch ", epoch, " : ", loss)



            Z1, A1, Z2, A2, Z3, A3 = forwardProp(x)

            backProp(Z1, A1, Z2, A2, Z3, A3, trainingAnswer, x)

            count += 1


    
# Processing and Normalizing Equations
def normalize(dataArray):
    '''
    A normalization function that normalizes all of the data
    in a 1D array between -1 and 1
       
    Args: 
    dataArray: A 1D numpy Array
       
    Returns: 1D array of Normalized Data'''
    
    maximum= np.amax(dataArray)
    minimum = np.amin(dataArray)

    returnArray = np.zeros(dataArray.size)

    for i in range(dataArray.size):
        returnArray[i] = 2*((dataArray[i] - minimum)/(maximum - minimum))-1

    return returnArray

#Loss Function: Binary Cross Entropy

def lossFunction(corrVal, probability):
    return -(corrVal*math.log(probability) + (1 - corrVal)*math.log(1-probability))

# Derivative of the Loss Function:
    
def deriv_lossFunction(corrVal, probability):
    return -corrVal/probability + (1-corrVal)/(1-probability)

# Activation Function:

def sigmoid(val):
    '''My chosen activation function for this project'''
    return 1/(1+math.exp(-val))

#Derivative of the Activation Function:

def deriv_sigmoid(val):
    return sigmoid(val) * (1-sigmoid(val))