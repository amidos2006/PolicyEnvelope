import math
import random
import os
from Chromosome import Chromosome

class Cell:
    def __init__(self, dim, numberOfFit, popSize):
        self._pop = []
        self._fit = [None] * numberOfFit
        self._popSize = popSize
        self._dim = dim
        return

    def getInfeasibleChromosomes(self):
        self._pop.sort(key=lambda c: c.getConstraints())
        return self._pop

    def getFeasibleChromosomes(self, avoid=-1):
        result = []
        for i in range(0,len(self._fit)):
            if not self._fit[i] == None and not i == avoid:
                result.append(self._fit[i])
        return result

    def assignChromosome(self, c): 
        # Not meet constraints
        if c.getConstraints() < 1:
            if len(self._pop) >= self._popSize:
                self._pop.sort(key=lambda c: c.getConstraints())
                del self._pop[0]
            self._pop.append(c)
        else:
            if self._fit[c.getPopulationType()] == None:
                self._fit[c.getPopulationType()] = c
            else:
                # feasible = self.getFeasibleChromosomes(c.getPopulationType())
                old = self._fit[c.getPopulationType()].getFitness()
                new = c.getFitness()
                if new > old or (new == old and random.random() < 0.5):
                    self._fit[c.getPopulationType()] = c

    def rankSelection(self, population):
        ranks = [0] * len(population)
        for i in range(0, len(ranks)):
            ranks[i] = i + 1
        total = sum(ranks)
        for i in range(1, len(ranks)):
            ranks[i] = (ranks[i] + ranks[i - 1]) / float(total)
        randomValue = random.random()
        for i in range(0, len(ranks)):
            if randomValue <= ranks[i]:
                return population[i]
        return population[-1]

    def getChromosome(self, fPercentage=0.5):
        feasible = self.getFeasibleChromosomes()
        infeasible = self.getInfeasibleChromosomes()
        if len(infeasible) == 0 or (len(feasible) > 0 and random.random() < fPercentage):
            return random.choice(feasible)
        return self.rankSelection(infeasible)

    def writeCell(self, filePath):
        feasible = self.getFeasibleChromosomes()
        infeasible = self.getInfeasibleChromosomes()
        for c in feasible:
            index = self._fit.index(c)
            c.writeChromosome(filePath + "feasible_" + str(c.getPopulationType()) + ".txt", self.getFeasibleChromosomes(index))
        for i in range(0, len(infeasible)):
            infeasible[i].writeChromosome(filePath + "infeasible_" + str(i) + ".txt")

class MapElite:
    def __init__(self):
        self._map = {}

    def initializeMap(self, width, height, gameInfo, lvlPercentage, initializationSize):
        chromosomes = []
        for i in range(initializationSize):
            temp = Chromosome(width, height, gameInfo)
            temp.randomInitialize(lvlPercentage)
            chromosomes.append(temp)
        return chromosomes

    def getCells(self):
        cells = []
        for key in self._map:
            cells.append(self._map[key])
        return cells

    def getNextGeneration(self, generationSize, inbreed, crossover, mutation):
        # ! seems didn't update chromosomes 
        chromosomes = [] 
        cells = self.getCells()
        for i in range(generationSize):
            cell1 = cells[random.randrange(0, len(cells))]
            cell2 = cells[random.randrange(0, len(cells))]
            if random.random() < inbreed:
                cell2 = cell1
            p1 = cell1.getChromosome()
            p2 = cell2.getChromosome()
            if random.random() < crossover:
         
                child = p1.crossover(p2)
                if random.random() < mutation:
                    child = child.mutate()
            else:
                child = p1.mutate()

            chromosomes.append(child)
        return chromosomes

    def updateMap(self, chromosomes, numberOfFit, popSize):
        for c in chromosomes:
            dims = c.getDimensions()
            key = str(dims[0])
            for i in range(1, len(dims)):
                key += "," + str(dims[i])
            if not key in self._map:
                self._map[key] = Cell(key, numberOfFit, popSize)
            self._map[key].assignChromosome(c)

    def writeMap(self, filePath):
        cells = self.getCells()
        for c in cells:
            os.mkdir(filePath + c._dim + "/")
            c.writeCell(filePath + c._dim + "/")
