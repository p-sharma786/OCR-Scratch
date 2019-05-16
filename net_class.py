import numpy as np
@np.vectorize

def sigmoid(x):
    tmp = 1/(1+np.e**-x)
    return tmp;

activation_function = sigmoid


class NeuralNetwork:
    #Constructor for initializing parameters of neural net
    def __init__(self,numInNodes,numOutNodes,numHiddenNodes,learningRate,bias = None):
        self.numInNodes = numInNodes
        self.numOutNodes = numOutNodes
        self.numHiddenNodes = numHiddenNodes
        self.learningRate = self.learningRate
        self.bias = bias
        # initializing Random Weights
        self.create_weight_matrices()



    def create_weight_matrices(self):
    	#a function to initialize the weights of two layered multiperceptron
        

        biasNode = 1 if self.bias else 0

        radHidden = 1/np.sqrt(self.numInNodes)
        radOuter = 1/np.sqrt(self.numHiddenNodes)
        
        self.weightsInHiddenLayer = np.random.randn(numHiddenNodes,numInNodes)*radHidden
        self.weightsInOuterLayer = np.random.randn(numOutNodes,numHiddenNodes)*radOuter
    
    #backpropogation - single iteration
    def back_propogation(self, inputVector, targetVector):
    '''targetVector is the vector using which the error will be clculated
       it contains the correct output value of the training dataset       ''' 

        biasNode = 1 if self.bias else 0

        if self.bias:
            #adding bias node at end
            inputVector = np.concatenate((inputVector,[self.bias]))
            
        #one time feedforward propogation for calculating error
        inputVector = np.array(inputVector,ndim=2).T
        targetVector = np.array(targetVector,ndim=2).T

        outputVector1 = np.dot(self.weightsInHiddenLayer,inputVector)
        outputVector1Final = activation_function(outputVector1)

        if self.bias:
            outputVector1Final = np.concatenate((outputVector1Final, [[self.bias]]))

        outputVector2 = np.dot(self.weightsInOuterLayer,outputVector1Final)
        outputVector2Final = activation_function(outputVector2)

       #calculating the errors(outer)
       outputErrors = targetVector - outputVector2Final

       #updating the weights
       tmp = outputErrors * outputVector2Final *(1.0 - outputVector2Final)
       tmp = self.learningRate * np.dot(tmp , outputVector1Final.T)
       self.weightsInOuterLayer += tmp

       #calculating the errors(inner)
       innerErrors = np.dot(self.weightsInOuterLayer.T, outputErrors)

       #updating the weights
       tmp = innerErrors *outputVector1Final * (1.0 - outputVector1Final)
       if self.bias:
            tmp2 = np.dot(tmp, inputVector.T)[:-1,:] #last element cutting
       else:
            tmp2 = np.dot(tmp,inputVector.T)

       self.weightsInHiddenLayer+= self.learningRate * tmp2  
    


        #
    def run(self, inputVector):
        # input could be tuple,list or ndarray

        if self.bias:
            # adding bias node at the end of input vector
            inputVector = np.concatenate((inputVector, [1]))

        outputVector1 = np.array(inputVector, ndim=2).T
        outputVector1 = np.dot(self.weightsInHiddenLayer, inputVector)
        outputVector1 = activation_function(outputVector1)

        if self.bias:
            outputVector1 = np.concatenate((outputVector1,[[1]]))

        outputVector2 = np.dot(self.weightsInOuterLayer,outputVector1)
        outputVector2 = activation_function(outputVector1)

        return outputVector2

    def train(self,dataArray,labelsOneHotArray,epochs=1,intermediateResults=False):
        intermediateWeights = []
        for epoch in range(epochs):
            for i in range(len(dataArray)):
                self.back_propogation(dataArray[i],labelsOneHotArray[i])

            if intermediateResults:
                intermediateWeights.append((self.weightsInHiddenLayer.copy(),self.weightsInOuterLayer.copy()))

        return intermediateWeights

    def evaluate(self,data,labels):
        numCorrect, numWrong = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            resMax = res.argmax()
            if resMax == labels[i]:
                numCorrect +=1
            else:
                numWrong +=1
        return numCorrect, numWrong







        