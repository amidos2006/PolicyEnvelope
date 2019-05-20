import math
import random
import numpy as np
from Global import *

""" 
NN: Neural Network
TS: Tree Search 
"""

class Chromosome:
    def __init__(self, width, height, gameInfo):
        self._genes = []
        for y in range(0, height):
            self._genes.append([])
            for x in range(0, width):
                self._genes[y].append(0)
        self._gameInfo = gameInfo
        # self._gameInfo = ray.get(gameInfo)

        self._results = {}
        self._results["NN"] = {}
        self._results["NN"]["win"] = []
        self._results["NN"]["score"] = []
        self._results["NN"]["time"] = []
        self._results["TS"] = {}
        self._results["TS"]["win"] = []
        self._results["TS"]["score"] = []
        self._results["TS"]["time"] = []

    def randomInitialize(self, lvlPercentage):
        totalIndex = 0
        locations = self._gameInfo.getLocations(self._genes, 0)  # Get all locations in genes
        random.shuffle(locations)

        nonStatisfying = self._gameInfo.getNonSatisfyMinMax(self._genes)
        while len(nonStatisfying) > 0:
            index = random.choice(nonStatisfying)
            pos = locations[totalIndex]
            self._genes[pos[1]][pos[0]] = index
            totalIndex += 1
            nonStatisfying = self._gameInfo.getNonSatisfyMinMax(self._genes)

        while totalIndex < lvlPercentage * len(self._genes) * len(self._genes[0]):
            pos = locations[totalIndex]
            self._genes[pos[1]][pos[0]] = self._gameInfo.getRandomValue(self._genes)
            totalIndex += 1

    def clone(self):
        clone = Chromosome(len(self._genes[0]), len(self._genes), self._gameInfo)
        for y in range(0, len(self._genes)):
            for x in range(0, len(self._genes[y])):
                clone._genes[y][x] = self._genes[y][x]
        return clone

    def getLevel(self):
        result = ""
        borderChar = self._gameInfo.getIndexToChar(self._gameInfo.getBorder())

        for j in range(0, len(self._genes[0]) + 2):
            result += borderChar
        result += "\n"

        lines = self.__str__().split("\n")
        for l in lines:
            if len(l.strip()) == 0:
                continue
            result += borderChar + l + borderChar + "\n"

        for j in range(0, len(self._genes[0]) + 2):
            result += borderChar

        return result

    def crossover(self, c):
        child = self.clone()
        x1 = random.randrange(0, len(child._genes[0]))
        y1 = random.randrange(0, len(child._genes))
        x2 = random.randrange(0, len(child._genes[0]))
        y2 = random.randrange(0, len(child._genes))
        if x1 > x2:
            temp = x1
            x1 = x2
            x2 = temp
        if y1 > y2:
            temp = y1
            y1 = y2
            y2 = temp
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                child._genes[y][x] = c._genes[y][x]
        return child

    def mutate(self):
        c = self.clone()
        x = random.randrange(0, len(c._genes[0]))
        y = random.randrange(0, len(c._genes))
        value = self._gameInfo.getRandomValue(self._genes)
        c._genes[y][x] = value
        return c

    def runAlgorithms(self, times=20):  # measure performance on NN and TS
        
        if self.getConstraints() < 1:
            return
        level = self.getLevel()

        res1 = self._gameInfo.runNN(level, times)
        self._results["NN"]["win"] = res1[0]
        self._results["NN"]["score"] = res1[1]
        self._results["NN"]["time"] = res1[2]
        
        
        res2 = self._gameInfo.runTS(level, times)
        self._results["TS"]["win"] = res2[:, 0]
        self._results["TS"]["score"] = res2[:, 1]
        self._results["TS"]["time"] = res2[:,2]
            
        # print(self._results)

    def getResults(self):
        return self._results
    
    def getConstraints(self):
        minMaxError = len(self._gameInfo.getNonSatisfyMinMax(self._genes))
        dijsktraError = self._gameInfo.getNonConnected(self._genes)
        return 1 / (math.log(minMaxError + dijsktraError + 1) + 1)


    def getFitness(self):
        return self._gameInfo.getEntropy(self._genes)
        # print(prob)
        # return -prob*math.log2(prob)

    # ! Need change to simplicity metrics fitness = (0.2*(1-H(x))) + (0.8*(1-x_hat))
    # def getFitness(self, pop):
    #     histogram = self._gameInfo.getHistogram(self._genes)
    #     minValue = 1
    #     for p in pop:
    #         tempHisto = p._gameInfo.getHistogram(p._genes)
    #         total = 0
    #         for i in range(0,len(histogram)):
    #             m = max(histogram[i], tempHisto[i])
    #             value = 0
    #             if m != 0:
    #                 value = abs(histogram[i] - tempHisto[i]) / float(m)
    #             total += value
    #         total /= float(len(histogram))
    #         if total < minValue:
    #             minValue = total
    #     return minValue

    def getPopulationType(self):
        if(len(self._results["NN"]["win"]) == 0 or len(self._results["TS"]["win"]) == 0):
            return int(0)
        nnWin = max(self._results["NN"]["win"])
        tsWin = max(self._results["TS"]["win"])
        return int(nnWin) + int(2 * tsWin)

    def getDimensions(self):
        maxScore = self._gameInfo.getMaxScore(self._genes) + 1
        maxTime = 500
        r = np.linspace(0,1,7)

        nnWin = self._results["NN"]["win"]
        tsWin = self._results["TS"]["win"]
        nnScore = self._results["NN"]["score"]
        nnTime = self._results["NN"]["time"]
        tsScore = self._results["TS"]["score"]
        tsTime = self._results["TS"]["time"]
        

        
        if len(nnWin)==0:
            nnWin = np.array([0])
            tsWin = np.array([0])
            nnScore = np.array([0])
            nnTime = np.array([maxTime])
            tsScore = np.array([0])
            tsTime = np.array([maxTime])
         
        
        if np.any(nnWin):
            nnScore = np.max(nnScore[np.where(nnWin!=0)]+1)/maxScore
            nnTime = np.min(nnTime[np.where(nnWin!=0)])/maxTime
        else:
            nnScore = np.mean(nnScore+1)/maxScore
            nnTime = np.mean(nnTime)/maxTime
        if np.any(tsWin):
            tsScore = np.max(tsScore[np.where(tsWin!=0)]+1)/maxScore
            tsTime = np.min(tsTime[np.where(tsWin!=0)])/maxTime
        else:
            tsScore = np.mean(tsScore+1)/maxScore
            tsTime = np.mean(tsTime)/maxTime   

        raw = [nnScore,tsScore,nnTime,tsTime]
        
        
        return  np.append([np.max(nnWin), np.max(tsWin)], np.piecewise(raw, [raw>=n for i,n in enumerate(r[:-1])],np.arange(0,6)))

    def __str__(self):
        result = ""
        for y in range(0, len(self._genes)):
            for x in range(0, len(self._genes[y])):
                result += self._gameInfo.getIndexToChar(self._genes[y][x])
            result += "\n"
        return result[:-1]

    def writeChromosome(self, filepath, pop=None):
        with open(filepath, "w+") as f:
            f.write("Type: " + str(self.getPopulationType()) + "\n")
            f.write("Constraints: " + str(self.getConstraints()) + "\n")
            if pop != None:
                record = self._gameInfo.getRecords(self._genes)
                f.write("Fitness: " + str(self.getFitness()) + "\n")
                f.write("NN win: "+ ",".join(map(str, self._results["NN"]["win"])) + "\n")
                f.write("TS win: "+ ",".join(map(str, self._results["TS"]["win"])) + "\n")
                f.write("NN score: "+ ",".join(map(str, self._results["NN"]["score"])) + "\n")
                f.write("TS score: "+ ",".join(map(str, self._results["TS"]["score"])) + "\n")
                f.write("NN time: "+ ",".join(map(str, self._results["NN"]["time"])) + "\n")
                f.write("TS time: "+ ",".join(map(str, self._results["TS"]["time"])) + "\n")
                f.write("emptyTiles: " + str(record[0]) + "\n")
                f.write("nearestKey: " + str(record[1]) + "\n")
                f.write("nearestDoor: " + str(record[2]) + "\n")
                f.write("longestPath: " + str(record[3]) + "\n")
                f.write("numConnectedComponent: " + str(record[4]) + "\n")
                f.write("maxScore: " + str(self._gameInfo.getMaxScore(self._genes)) + "\n")
                nnScore = (1.0 * max(self._results["NN"]["score"])) / max(1, self._gameInfo.getMaxScore(self._genes))
                tsScore = (1.0 * max(self._results["TS"]["score"])) / max(1, self._gameInfo.getMaxScore(self._genes))
                f.write("NN-TS Score: " + str(nnScore)  + " - " + str(tsScore) + "\n")
            else:
                f.write("Fitness: 0\n")
                f.write("NN-TS Score: 0 - 0\n")
            f.write("Level:\n")
            f.write(self.getLevel())
