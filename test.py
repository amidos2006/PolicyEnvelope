from Global import *
from MapElite import MapElite
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing import Pool
import numpy as np
import argparse
import uuid

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

# TODO: adding parallel

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def generator(batchSize):
    worker_id = str(uuid.uuid1())
    map = MapElite()
    chs = map.initializeMap(width, height, Zelda(testPath + worker_id + "/level.txt", testPath + worker_id + "/result.txt"), lvlPercentage, batchSize)
    for i in range(0, iterations + 1):
        for c in chs:
            c.runAlgorithms(numOfTests)  # measure each chromosome's performance and create positiion in feature space
        map.updateMap(chs, numberOfFit, populationSize)
        writeIteration(resultPath + worker_id + "/", i, map)
        updateIterationResults(resultPath + worker_id + "/", i, map)
        deleteIteration(resultPath + worker_id + "/", i-1)
        chs = map.getNextGeneration(batchSize, inbreed, crossover, mutation)
        print('chromesome lenght: ',len(chs))

def split_batchSize(batchSize, split):
    remain = batchSize % split
    rest = batchSize // split
    return [rest]*split + [remain]

def args_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', help='worker number',type=int ,default=mp.cpu_count())
    return parser

if __name__ == "__main__":
    print("------------------ generate ------------------")
    args = args_parse().parse_args()
    worker_count = args.n
    batchSizeInWorker = split_batchSize(batchSize, worker_count)
    p = MyPool(worker_count)
    p.map(generator, batchSizeInWorker)
    p.close()
    p.join()
