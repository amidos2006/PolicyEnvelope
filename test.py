from Global import *
from MapElite import MapElite

width = 13
height = 9
iterations = 10
inputFile = "test/level.txt"
outputFile = "test/result.txt"
resultPath = "result/"
lvlPercentage = 0.5
batchSize = 100
numberOfFit = 4
populationSize = 20
numOfTests = 20
inbreed = 0.5
crossover = 0.7
mutation = 0.3

map = MapElite()
chs = map.initializeMap(width, height, Zelda(inputFile, outputFile), lvlPercentage, batchSize)
for i in range(0, iterations + 1):
    for c in chs:
        c.runAlgorithms(numOfTests)
    map.updateMap(chs, numberOfFit, populationSize)
    writeIteration(resultPath, i, map)
    updateIterationResults(resultPath, i, map)
    deleteIteration(resultPath, i-1)
    chs = map.getNextGeneration(batchSize, inbreed, crossover, mutation)
