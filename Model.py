#import libraries
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import copy

# Algorithms:
class Model:

    weight1 = np.zeros((11, 5))        
    
    bias1 = np.zeros((5,1))

    weight2 = np.zeros((5, 1))

    bias2 = np.zeros((1, 1))

    output = np.zeros((1, 1))

    outputBias = np.zeros((1, 1))                   

    tempWeight1 = np.zeros((11, 5))        
    
    tempBias1 = np.zeros((5,1))

    tempWeight2 = np.zeros((5, 1))

    tempBias2 = np.zeros((1, 1))

    tempOutput = np.zeros((1, 1))

    tempOutputBias = np.zeros((1, 1))   
    


   


    #initializing weights and biases:
    
    def __init__(self):
        '''Initialize all weights and biases:
        2 Hidden Layer
        1 Output Layer'''

        self.weight1[:] = (np.random.rand(11, 5) * 2) - 1

        self.bias1 = (np.random.randn(5,1)*2)-1
    
        self.weight2[:] = (np.random.rand(5, 1) * 2) - 1
    
        self.bias2 = (np.random.randn(1,1)*2)-1
    
        self.output[:] = (np.random.rand(1, 1) * 2) - 1
    
        self.outputBias[:] = (np.random.rand(1, 1) * 2) - 1

        self.tempWeight1 = copy.copy(self.weight1)
        self.tempBias1 = copy.copy(self.bias1)

        self.tempWeight2 = copy.copy(self.weight2)
        self.tempBias2 = copy.copy(self.bias2)

        self.tempOutput = copy.copy(self.output)
        self.tempOutputBias = copy.copy(self.outputBias)





        

        
    @staticmethod
    def calculate(self, dataPoint):
        Z1 = self.weight1.dot(dataPoint) + self.bias1
        A1 = map(self.sigmoid, Z1)

        Z2 = self.weight2.T.dot(A1) + self.bias2
        A2 = map(self.sigmoid, Z2)

        Z3 = self.output.T.dot(A2) + self.outputBias
        A3 = map(self.sigmoid, Z3)

        return A3


    def eval(self, val_set, val_set_answer):
            
            transfer = 0

            for i in val_set:
                Z1 = self.weight1.T.dot(i) + self.bias1
                A1 = map(self.sigmoid, Z1)

                Z2 = self.weight2.T.dot(A1) + self.bias2
                A2 = map(self.sigmoid, Z2)

                Z3 = self.output.T.dot(A2) + self.outputBias
                A3 = map(self.sigmoid, Z3)

                transfer += self.lossFunction(A3, i)

            return transfer/val_set[0].size
    

        
    # BackPropogation Algorithm Begins:

    # Forward Prop:
    
    def forwardProp(self, trainingData):
        Z1 = self.weight1.T.dot(trainingData) + self.bias1
        A1 = np.vectorize(self.sigmoid)(Z1)

        Z2 = self.weight2.T.dot(A1) + self.bias2
        A2 = np.vectorize(self.sigmoid)(Z2)

        Z3 = self.output.T.dot(A2) + self.outputBias
        A3 = np.vectorize(self.sigmoid)(Z3)

        return Z1, A1, Z2, A2, Z3, A3
    

    # Back Prop:

    def backProp(self, Z1, A1, Z2, A2, Z3, A3, answer, input):
        
        val = -1
        
        third2 = self.deriv_lossFunction(answer, A3)
        third1 = self.deriv_sigmoid(Z3)
    
        
        temp3 = A2.dot(third2 * third1)

        temp3Bias = third2 * third1

        temp2 = A1.dot(((third2 * third1).dot(self.output)).dot(A2.dot((A2 * val) + 1)))
    
        temp2Bias = ((third2 * third1).dot(self.output)).dot(A2.dot((A2 * val) + 1))

        temp1 = input.dot((self.weight2.dot(((third2 * third1).dot(self.output)).dot(A2.dot((A2*-1) + 1)))).dot(A1.dot((A1*-1) + 1)))
    
        temp1Bias = (self.weight2.dot(((third2 * third1).dot(self.output)).dot(A2.dot((A2*-1) + 1)))).dot(A1.dot((A1*-1) + 1))
    

        tempOutput = tempOutput - 0.01 * temp3
    
        tempOutputBias = self.tempOuputBias - 0.01 * temp3Bias
    
        tempWeight2 = tempWeight2 - 0.01 * temp2
    
        tempBias2 = tempBias2 - 0.01 * temp2Bias

        tempWeight1 = tempWeight1 - 0.01 * temp1

        tempBias1 = tempBias1 - 0.01 * temp1Bias



    def training(self, trainingData, trainingAnswer,  val_set, val_set_answer):

        epoch = 0
        count = 0
        loss = float()

        for x in trainingData:
            
            if count > 27:

                self.weight1 = self.tempWeight1
                    
                self.weight2= self.tempWeight2

                self.output = self.tempOutput

                self.outputBias = self.tempOutputBias

                self.bias1 = self.tempBias1

                self.bias2 = self.tempBias2

                self.outputBias = self.tempOutputBias

                loss = eval(val_set, val_set_answer)

                epoch += 1

                count = 0

                print("Loss on epoch ", epoch, " : ", loss)



            Z1, A1, Z2, A2, Z3, A3 = self.forwardProp(x.T)

            self.backProp(Z1, A1, Z2, A2, Z3, A3, trainingAnswer, x)

            count += 1


        # Processing and Normalizing Equations
    def normalize(self, dataArray):
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

    @staticmethod
    def lossFunction(corrVal, probability):
        return -(corrVal*math.log(probability) + (1 - corrVal)*math.log(1-probability))

    # Derivative of the Loss Function:
    
    @staticmethod
    def deriv_lossFunction(corrVal, probability):
        return -corrVal/probability + (1-corrVal)/(1-probability)

    # Activation Function:

    @staticmethod
    def sigmoid(val):
        '''My chosen activation function for this project'''
        return 1/(1+np.power(np.e, val))

    #Derivative of the Activation Function:

    
    def deriv_sigmoid(self, val):
        return self.sigmoid(val) * (1-self.sigmoid(val))


    
