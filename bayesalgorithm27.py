from bayesian27 import DataHandler
import math

class bayes:

	def __init__(self, x_train, y_train):
		self.dh = DataHandler(x_train, y_train)

	def calculateProbability(self, x, mean, stdev):
		if(stdev != 0):
			exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
			return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
		return 100 # jak odchylenie standardowe = 0 to jest 100% zgodnosci
	 
	def calculateClassProbabilities(self, summaries, inputVector):
		probabilities = {}
		for classValue, classSummaries in summaries.items():
			probabilities[classValue] = 1
			for i in range(len(classSummaries)):
				mean, stdev = classSummaries[i]
				x = inputVector[i]
				probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
		return probabilities
			
	def predict(self, summaries, inputVector):
		probabilities = self.calculateClassProbabilities(summaries, inputVector)
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.items():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel
	 
	def getPredictions(self, summaries, testSet):
		predictions = []
		for i in range(len(testSet)):
			result = self.predict(summaries, testSet[i])
			predictions.append(result)
		return predictions
	 
	def getAccuracy(self, testSetPredictions, predictions):
		correct = 0
		for i in range(len(testSetPredictions)):
			if testSetPredictions[i] == predictions[i]:
				correct += 1
		return (correct/float(len(testSetPredictions))) * 100.0
 
