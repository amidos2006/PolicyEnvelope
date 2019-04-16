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

width = 11
height = 7
iterations = 10
testPath = "test/"
inputFile = "test/level"
outputFile = "test/result"
resultPath = "result/"
lvlPercentage = 0.5
batchSize = 100   
numberOfFit = 4
populationSize = 20
numOfTests = 20
inbreed = 0.5
crossover = 0.7
mutation = 0.3
initializationSize = 1000
generationSize = 10

# TODO: adding parallel

def runAlgorithm(c, numOfTests):
    print("runAlgorithm")
    pdf.set_trace()
    # c.runAlgorithms(numOfTests)
    
def test(a):
    print("test")
    
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def generator(batchSize, initializationSize, numOfTests, generationSize, iterations):
    worker_id = time.asctime() + str(uuid.uuid1())
    map = MapElite()
    chs = map.initializeMap(width, height, Zelda(testPath + worker_id + "/level.txt", testPath + worker_id + "/result.txt"), lvlPercentage, initializationSize)
    for i in range(0, iterations + 1):
        # p = Pool(12)
        # batchSizeInWorker = split_batchSize(batchSize, chs, nums)
        # print("iteration")
        # # p.map(runAlgorithm, list(zip(chs, [numOfTests]*initializationSize)))
        # p.map(test,chs[:12])
        # p.close()
        # p.join()
        # p.starmap(Chromosome.runAlgorithms, zip(chs, [numOfTests]*initializationSize))
        # p.close()
        # p.join()
        # print("join")
        # p = MyPool(12)
        for c in chs:
            c.runAlgorithms(numOfTests)
        #     p.map(Chromosome.runAlgorithms,(chs,numOfTests,))
        # p.close()
        # p.join()
        map.updateMap(chs, numberOfFit, populationSize)
        writeIteration(resultPath + worker_id + "/", i, map)
        updateIterationResults(resultPath + worker_id + "/", i, map)
        deleteIteration(resultPath + worker_id + "/", i-1)
        chs = map.getNextGeneration(generationSize, inbreed, crossover, mutation)
        print('chromesome lenght: ',len(chs))


def split_batchSize(batchSize, split):
    remain = batchSize % split
    rest = batchSize // split
    return [rest]*split + [remain]

# def split_batchSize(batchSize,split ,chs, nums):
#     combo = zip(chs, [nums]*batchSize)
#     remain = batchSize % split
#     rest = batchSize // split
    
#     return [rest]*split + [remain]

def args_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', help='worker number',type=int ,default=mp.cpu_count())
    parser.add_argument('--genSize', help='generation size',type=int ,default=1)
    parser.add_argument('--numTests', help='number of tests',type=int ,default=10)
    parser.add_argument('--init', help='initialization size',type=int ,default=100)
    parser.add_argument('--i', help='iteration size',type=int ,default=100)
    return parser

if __name__ == "__main__":
    print("------------------ generate ------------------")
    args = args_parse().parse_args()
    worker_count = args.n
    # batchSizeInWorker = split_batchSize(batchSize, worker_count)
    # params = zip(batchSizeInWorker,[args.i]*worker_count,[args.numTests]*worker_count ,[args.genSize]*worker_count)
    # p = MyPool(worker_count)
    # p.starmap(generator, params)
    # p.close()
    # p.join()
    generator(batchSize, args.init, args.numTests, args.genSize, args.i)
