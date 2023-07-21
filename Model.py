#import libraries
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import copy

# Algorithms:
class Model:

    #Initialize Weight Matrix Sizes

    weight1 = np.zeros((21, 5))        
    
    bias1 = np.zeros((5,1))

    

    output = np.zeros((5, 1))

    outputBias = np.zeros((1, 1))   



    tempWeight1 = np.zeros((21, 5))        
    
    tempBias1 = np.zeros((5,1))

   

    tempOutput = np.zeros((5, 1))

    tempOutputBias = np.zeros((1, 1))  


    lossData = np.zeros(150)

    # Adam Optimization Value Initialization

    Mt2 = 0.00
    Vt2 = 0.00

    Mt2B = 0.00
    Vt2B = 0.00

    Mt1 = 0.00
    Vt1 = 0.00

    Mt1B = 0.00
    Vt1B = 0.00
    
    A1Rand = np.zeros((5, 1))
  
   
    
    def __init__(self):
        '''Initialize all weights and biases values:
        2 Hidden Layer
        1 Output Layer'''

        self.weight1[:] = self.xavierInit(21, 5)

        self.bias1 = self.xavierInit(5, 1)
    
        # 2*(np.random.rand(11, 5))-1
    
        self.output[:] = self.xavierInit(5, 1)
    
        self.outputBias[:] = self.xavierInit(1, 1)


        self.tempWeight1 = copy.copy(self.weight1)
        self.tempBias1 = copy.copy(self.bias1)

        

        self.tempOutput = copy.copy(self.output)
        self.tempOutputBias = copy.copy(self.outputBias)


    # Evaluates the model using a validation set

    def evaluation(self, val_set, val_set_answer):
            
            transfer = 0

            for i in range(val_set[:, 0].size): 

                Z1 = (self.weight1).T.dot(val_set[i, :]).reshape(5, 1) + self.bias1
                A1 = np.vectorize(self.ReLu)(Z1)
                
                Z3 = (self.output).T.dot(A1) + self.outputBias
                A3 = np.vectorize(self.sigmoid)(Z3)

                val = val_set_answer[i, :]

                hold = self.lossFunction(val_set_answer[i, :].reshape(1, 1), A3)
                transfer+= hold

            return transfer/val_set[:, 0].size
    

        
    # BackPropogation Algorithm Begins:

    # Forward Prop:
    
    def forwardProp(self, trainingData):

        Z1 = np.dot(self.weight1.T, trainingData).reshape(5, 1) + self.bias1
        A1 = np.vectorize(self.ReLu)(Z1)

        Z3 = self.output.T.dot(A1) + self.outputBias
        A3 = np.vectorize(self.sigmoid)(Z3)


        return Z1, A1, Z3, A3
    

    # Back Prop:

    def backProp(self, Z1, A1, Z3, A3, answer, input):

        neuronGradient3 = np.dot(self.deriv_lossFunction(answer.reshape(1,1), A3), self.deriv_sigmoid(Z3))

        temp3 = np.dot(A1, neuronGradient3)

        neuronGradient1 = np.dot((self.deriv_ReLu(Z1)),  np.dot(self.tempOutput, neuronGradient3))
    
        temp1 = np.dot(input.reshape(21, 1), neuronGradient1.T)


        self.tempOutput = self.tempOutput - 0.01 * self.Adam(temp3, "2")

        self.tempOutputBias = self.tempOutputBias - 0.01 * self.Adam(neuronGradient3, "2B")

        self.tempWeight1 = self.tempWeight1 - 0.01 * self.Adam(temp1, "1")

        self.tempBias1 = self.tempBias1 - 0.01 * self.Adam(neuronGradient1, "1B")


    # A training function where frontProp and backProp are called to configure weight and bias values

    def training(self, trainingData, trainingAnswer,  val_set, val_set_answer):

        epoch = 0
        count = 0
        loss = float()


        for i in range((trainingData[:, 0].size)):
            
            if count > 10:
                

                self.weight1 = self.tempWeight1
                    
                self.output = self.tempOutput

                self.outputBias = self.tempOutputBias

                self.bias1 = self.tempBias1


                loss = self.evaluation(val_set, val_set_answer)

                self.lossData[epoch] = loss

                epoch += 1

                count = 0

                print("Loss on epoch ", epoch, " : ", loss)



            Z1, A1, Z3, A3 = self.forwardProp(trainingData[i].T)

            self.backProp(Z1, A1, Z3, A3, trainingAnswer[i], trainingData[i].T)

            count+=1

        plt.plot(self.lossData)
        plt.savefig("myGraph.png")
       


    # Processing and Standardizing Equations
    def standardize(self, dataArray):
        
        
        maximum= np.amax(dataArray)
        minimum = np.amin(dataArray)

        returnArray = np.zeros(dataArray.size)

        for i in range(dataArray.size):
            returnArray[i] = ((dataArray[i] - minimum)/(maximum - minimum))

        return returnArray

    #Loss Function: Binary Cross Entropy

    @staticmethod
    def lossFunction(corrVal, probability):
        return -(corrVal * math.log(probability) + (1 - corrVal) * math.log(1-probability))

    # Derivative of the Loss Function:
    
    @staticmethod
    def deriv_lossFunction(corrVal, probability):
        return -corrVal/probability + (1-corrVal)/(1-probability)

    # Output Activation Function:

    def sigmoid(self, val):
        return 1/(1+np.power(np.e, -val))
         
    
    #Activation Function:

    def ReLu(self, input):
            return max(0, input)

    #Derivative of the Activation Function:
    
    def deriv_ReLu(self, input):

        if input.all() < 0: return 0
        else: return 1

  
    #Derivative of the output Activation Function:
    
    def deriv_sigmoid(self, val):
    
        return np.dot(self.sigmoid(val), (1-self.sigmoid(val)).T)
    
    # Xavier Weight Initialization:
    def xavierInit(self, inputs, outputs):
        return math.sqrt(6/(inputs + outputs))

    # Adam Optimization:
    def Adam(self, weightGradient, track):

        B1 = 0.9
        B2 = 0.999

        if track == "2":

            self.Mt2 = B1 * self.Mt2 + (1-B1) * weightGradient

            self.Vt2 = B2 * self.Vt2 + (1-B2) * weightGradient**2

            MT = self.Mt2 / (1-B1)

            VT = self.Vt2 / (1-B2)

            returnVal = MT / ((np.sqrt(VT.astype(np.float32))) + 0.00000001)

            return returnVal
        
        if track == "2B":
            self.Mt2B = B1 * self.Mt2B + (1-B1) * weightGradient

            self.Vt2B = B2 * self.Vt2B + (1-B2) * weightGradient**2

            MT = self.Mt2B / (1-B1)

            VT = self.Vt2B / (1-B2)

            returnVal = MT / ((np.sqrt(VT.astype(np.float32))) + 0.00000001)

            return returnVal
        
        
        if track == "1":
            self.Mt1 = B1 * self.Mt1 + (1-B1) * weightGradient

            self.Vt1 = B2 * self.Vt1 + (1-B2) * weightGradient**2

            MT = self.Mt1 / (1-B1)

            VT = self.Vt1 / (1-B2)

            returnVal = MT / ((np.sqrt(VT.astype(np.float32))) + 0.00000001)

            return returnVal
        
        if track == "1B":
            self.Mt1B = B1 * self.Mt1B + (1-B1) * weightGradient

            self.Vt1B = B2 * self.Vt1B + (1-B2) * weightGradient**2

            MT = self.Mt1B / (1-B1)

            VT = self.Vt1B / (1-B2)

            returnVal = MT / ((np.sqrt(VT.astype(np.float32))) + 0.00000001)

            return returnVal

    
    
