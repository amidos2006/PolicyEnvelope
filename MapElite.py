import math
import random
import os
from Chromosome import Chromosome

class Cell:
    def __init__(self, dim):
        # self._pop = []
        # self._fit = [None] * numberOfFit
        # self._popSize = popSize
        self.c = None
        self._dim = dim
        return

    def assignChromosome(self, c): 
        if self.c== None:
            self.c = c
        else:
            old = self.c.getFitness()
            new = c.getFitness()
            if new > old or (new == old and random.random() < 0.5):
                self.c = c

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
        # feasible = self.getFeasibleChromosomes()
        # infeasible = self.getInfeasibleChromosomes()
        # if len(infeasible) == 0 or (len(feasible) > 0 and random.random() < fPercentage):
        #     return random.choice(feasible)
        # return self.rankSelection(infeasible)
        return self.c

    def writeCell(self, filePath):
        c = self.c
        c.writeChromosome(filePath + "feasible_" + str(c.getPopulationType()) + ".txt", c)
        # for i in range(0, len(infeasible)):
        #     infeasible[i].writeChromosome(filePath + "infeasible_" + str(i) + ".txt")

class MapElite:
    def __init__(self):
        self._map = {}
        self.population = []

    def getPopulation(self):
        return self.population

    def initializeMap(self, width, height, gameInfo, lvlPercentage, initializationSize):
        chromosomes = []
        for _ in range(initializationSize):
            temp = Chromosome(width, height, gameInfo)
            temp.randomInitialize(lvlPercentage)
            chromosomes.append(temp)
        return chromosomes

    def getCells(self):
        cells = []
        for key in self._map:
            cells.append(self._map[key])
        return cells

    def getNextGeneration(self, generationSize, inbreed, crossover, mutation, eva=False):
        chromosomes = [] 
        if not eva:
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
        else:
            pop = self.population
            for i in range(generationSize):
                p1, p2 = random.choices(pop, k=2)
                if random.random() < inbreed:
                    p2 = p1
                if random.random() < crossover:
                    child = p1.crossover(p2)
                    if random.random() < mutation:
                        child = child.mutate()
                else:
                    child = p1.mutate()

                chromosomes.append(child)
        return chromosomes

    def updateMap(self, chromosomes):
        print("update")
        for c in chromosomes:
            if c.getConstraints() < 1:
                print("in constraints")
                self.population.append(c)
            else:
                dims = c.getDimensions()
                key = ",".join(str(d) for d in dims)
                if not key in self._map:
                    self._map[key] = Cell(key)
                self._map[key].assignChromosome(c)

    def writeMap(self, filePath):
        cells = self.getCells()
        for c in cells:
            os.mkdir(filePath + c._dim + "/")
            c.writeCell(filePath + c._dim + "/")
