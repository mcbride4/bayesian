import csv
import random
import numpy

class DataHandler:

    def __init__(self, x_train, y_train):
        self.predictions = []
        self.dataset = x_train
        self.predictions = y_train
        self.trainData = list(self.dataset)
        self.trainDataPredictions = list(self.predictions)
        self.testData = []
        self.testDataPredictions = []
        self.separatedByClass = {}
        self.stats = {}

    def loadCsv(self, filename):
        with open(filename, 'r') as csvfile:
            self.dataset = list(csv.reader(csvfile))
            for i in self.dataset:
                self.predictions.append(i.pop())
            self.__prepareDataset(len(self.dataset))
        return self.dataset

    def splitData(self, percentOfTestData):
        testDataLen = int(percentOfTestData * len(self.trainData))
        while len(self.testData) < testDataLen:
            rand = random.randrange(len(self.trainData))
            self.testData.append(self.trainData.pop(rand))
            self.testDataPredictions.append(self.trainDataPredictions.pop(rand))

    def separateByClass(self, data, predictions):
        for i in range(len(predictions)):
            self.__appendRow(data[i], predictions[i])

    def statsByClass(self):
        for classValue, instances in self.separatedByClass.items():
            self.stats[classValue] = self.__calculateStats(instances)

    def __prepareDataset(self, length):
        for i in range(length):
            self.dataset[i] = [float(x) for x in self.dataset[i]]

    def __appendRow(self, row, classValue):
        self.__createIfNotExists(classValue)
        self.separatedByClass[classValue].append(row)

    def __createIfNotExists(self, classValue):
        if (classValue not in self.separatedByClass):
            self.separatedByClass[classValue] = []

    def __calculateStats(self, data):
        stats = [(numpy.mean(attr), numpy.std(attr)) for attr in zip(*data)]
        return stats

