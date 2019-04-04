import math
import random

class Chromosome:
    def __init__(self, width, height, gameInfo):
        self._genes = []
        for y in range(0, height):
            self._genes.append([])
            for x in range(0, width):
                self._genes[y].append(0)
        self._gameInfo = gameInfo

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
        locations = self._gameInfo.getLocations(self._genes, 0)
        random.shuffle(locations)

        nonStatisfying = self._gameInfo.getNonSatisfyMinMax(self._genes)
        while len(nonStatisfying) > 0:
            index = random.choice(nonStatisfying)
            pos = locations[totalIndex]
            self._genes[pos[1]][pos[0]] = index
            totalIndex += 1
            nonStatisfying = self._gameInfo.getNonSatisfyMinMax(self._genes)

        while totalIndex < lvlPercentage * len(self._genes) * len(self._genes[0]):
            pos = locations[totalIndex];
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
                continue;
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

    def runAlgorithms(self, times=20):
        if self.getConstraints() < 1:
            return

        for i in range(0, times):
            temp = self._gameInfo.runNN(self.getLevel(), i)
            self._results["NN"]["win"].append(temp[0])
            self._results["NN"]["score"].append(temp[1])
            self._results["NN"]["time"].append(temp[2])
            temp = self._gameInfo.runTS(self.getLevel(), i)
            self._results["TS"]["win"].append(temp[0])
            self._results["TS"]["score"].append(temp[1])
            self._results["TS"]["time"].append(temp[2])

    def getConstraints(self):
        minMaxError = len(self._gameInfo.getNonSatisfyMinMax(self._genes))
        dijsktraError = self._gameInfo.getNonConnected(self._genes)
        return 1 / (math.log(minMaxError + dijsktraError + 1) + 1)

    def getFitness(self, pop):
        histogram = self._gameInfo.getHistogram(self._genes)
        minValue = 1
        for p in pop:
            tempHisto = p._gameInfo.getHistogram(p._genes)
            total = 0
            for i in range(0,len(histogram)):
                m = max(histogram[i], tempHisto[i])
                value = 0
                if m != 0:
                    value = abs(histogram[i] - tempHisto[i]) / float(m)
                total += value
            total /= float(len(histogram))
            if total < minValue:
                minValue = total
        return minValue

    def getPopulationType(self):
        if(len(self._results["NN"]["win"]) == 0 or len(self._results["TS"]["win"]) == 0):
            return int(0)
        nnWin = max(self._results["NN"]["win"])
        tsWin = max(self._results["TS"]["win"])
        return int(nnWin) + int(2 * tsWin)

    def getDimensions(self):
        return self._gameInfo.getDimensions(self._genes)

    def __str__(self):
        result = ""
        for y in range(0, len(self._genes)):
            for x in range(0, len(self._genes[y])):
                result += self._gameInfo.getIndexToChar(self._genes[y][x])
            result += "\n"
        return result[:-1]

    def writeChromosome(self, filepath, pop=None):
        f = open(filepath, "w")
        f.write("Type: " + str(self.getPopulationType()) + "\n")
        f.write("Constraints: " + str(self.getConstraints()) + "\n")
        if pop != None:
            f.write("Fitness: " + str(self.getFitness(pop)) + "\n")
            nnScore = (1.0 * max(self._results["NN"]["score"])) / max(1, self._gameInfo.getMaxScore(self._genes))
            tsScore = (1.0 * max(self._results["TS"]["score"])) / max(1, self._gameInfo.getMaxScore(self._genes))
            f.write("NN-TS Score: " + str(nnScore)  + " - " + str(tsScore) + "\n")
        else:
            f.write("Fitness: 0\n")
            f.write("NN-TS Score: 0 - 0\n")
        f.write("Level:\n")
        f.write(self.getLevel())
        f.close()