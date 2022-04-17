import numpy as np
import dataset

gain = 2
minCost = 0.01

trainingSet = dataset.trainingSet
testSet = dataset.testSet
outputDict = dataset.outputDict

def sigmoid (Z):
    return 1/(1+np.exp(-Z))

class NeuralNet:

    def __init__(self, n_in, n_hidden, n_out):
        # Network dimensions
        self.n_x = n_in
        self.n_h = n_hidden
        self.n_y = n_out

        # Parameters initialization
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        # self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        # self.b2 = np.zeros((self.n_y, 1))

    def read (self, trainingSet, outputDict):
        self.trainingSet = trainingSet
        self.outputDict = outputDict

        #inputs
        self.x = []

        #outputs
        self.trueValue = []

        for input in trainingSet:
                self.x.append(np.array(input[0]).flatten())

        self.x = np.array(self.x).T

        for output in trainingSet:
            self.trueValue.append(np.array(outputDict[output[1]]))

        self.trueValue = np.array(self.trueValue).T

    def report (self, caption):
        print ()
        print ('##################################################')
        print (f'       {caption} - LOSS: {self.cost()}')
        print ('##################################################')
        for itemIndex, item in enumerate (self.trainingSet):
            output = self.outputPrediction.T [itemIndex]
            value = self.trainingSet[itemIndex][1]
            print('')
            print (f' {value} --> {output}')

    def feedForward (self):

        self.layerPrediction = sigmoid(np.dot(self.W1, self.x))
        self.outputPrediction = sigmoid(np.dot(self.W2, self.layerPrediction))

    def cost(self):
        return np.sum((self.trueValue - self.outputPrediction)**2) / len(self.trueValue)

    def correct (self):
        weightsTensor = (self.W1, self.W2)
        bestWeightsIndex = 0
        bestRowIndex = 0
        bestColumnIndex = 0
        bestFactor = 1
        currentCost = self.cost ()
        lowestCost = currentCost

        for weightsIndex, weights in enumerate (weightsTensor):
            for rowIndex in range (weights.shape [0]):
                for columnIndex in range (weights.shape[1]):
                    for signedGain in (-gain, gain):
                        originalWeight = weights [rowIndex, columnIndex]
                        factor = 1 + signedGain * currentCost
                        weightsTensor [weightsIndex] [rowIndex, columnIndex] *= factor
                        self.feedForward()
                        cost = self.cost()
                        if cost < lowestCost:
                            lowestCost = cost
                            bestWeightsIndex = weightsIndex
                            bestRowIndex = rowIndex
                            bestColumnIndex = columnIndex
                            bestFactor = factor
                        weightsTensor [weightsIndex] [rowIndex, columnIndex] = originalWeight
        weightsTensor [bestWeightsIndex][bestRowIndex, bestColumnIndex ] *= bestFactor


    def train (self, trainingSet, minCost):
        self.read (trainingSet, outputDict)
        while True:
            self.feedForward()
            print (int (self.cost() * 100) * '*')

            print(f'LOSS: {self.cost()}')

            if self.cost() < minCost:
                self.report("Train")
                break

            self.correct()

    def predict (self, testSet):
        self.read(testSet, outputDict)
        self.feedForward()
        self.report("Predict")

print ("####################################################################")
print ("####################### BASIC NEURAL NETWORK - NUMPY ###############")
print ("####################################################################")
print ("")
print ("")
print ("-------------------------------------------------------------------------")
print ("                        Initialising network ........")
print ("-------------------------------------------------------------------------")
print ("")

input_rows = 3
input_columns = 3
output_n = 2

nn = NeuralNet(9, 3, 2)

print (f' Total input rows: {input_rows}')
print (f' Total input columns: {input_columns}')
print (f' Total output nodes: {input_rows}')

print ("")
print ("-------------------------------------------------------------------------")
print ("                        Training network ........")
print ("-------------------------------------------------------------------------")
print ("")

nn.train (trainingSet, minCost)

print ("")
print ("-------------------------------------------------------------------------")
print ("                        Predicting network ........")
print ("-------------------------------------------------------------------------")
print ("")

nn.predict (testSet)






