from Global import *
from MapElite import MapElite
from Chromosome import Chromosome
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing import Pool
import numpy as np
import argparse
import uuid
import time
import itertools
import pdb
import copy
from joblib import Parallel, delayed
import level_selector as ls

width = 11
height = 7
iterations = 10
testPath = "test/"
inputFile = "test/level"
outputFile = "test/result"
resultPath = "result/"
lvlPercentage = 0.1
batchSize = 100   
numberOfFit = 4
populationSize = 50
numOfTests = 20
inbreed = 0.5
crossover = 0.7
mutation = 0.3
initializationSize = 1000
generationSize = 10


def generator(checkpoint, initializationSize, numOfTests, generationSize, iterations):
    worker_id = time.asctime() + str(uuid.uuid1()) + " checkpoint_" +str(checkpoint) + " init_" + str(initializationSize) + " numTests_" + str(numOfTests) + " genSize_" + str(generationSize) + " i_" + str(iterations)

    map = MapElite()
    chs = map.initializeMap(width, height, Zelda(testPath + worker_id + "/level.txt", testPath + worker_id + "/result.txt", numOfTests), lvlPercentage, initializationSize)
    for i in range(0, iterations + 1):
        print("iteration: ", i)

        for c in chs:
            c.runAlgorithms(numOfTests)                

        print("join")
        map.updateMap(chs, numberOfFit, populationSize)
        if i%checkpoint == 0 or i == iterations:
            writeIteration(resultPath + worker_id + "/", i, map)
        updateIterationResults(resultPath + worker_id + "/", i, map) # update results in results file
        # deleteIteration(resultPath + worker_id + "/", i-1)
        chs = map.getNextGeneration(generationSize, inbreed, crossover, mutation)
        print('chromesome lenght: ',len(chs))


def split_batchSize(batchSize, split):
    remain = batchSize % split
    rest = batchSize // split
    return [rest]*split + [remain]


def args_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--genSize', help='generation size',type=int ,default=1)
    parser.add_argument('--numTests', help='number of tests',type=int ,default=10)
    parser.add_argument('--init', help='initialization size',type=int ,default=100)
    parser.add_argument('--i', help='iteration size',type=int ,default=100)
    parser.add_argument('--c', help='checkpoint',type=int ,default=100)
    return parser

if __name__ == "__main__":
    args = args_parse().parse_args()
    print("args: ", args)
    generator(args.c, args.init, args.numTests, args.genSize, args.i)
