import os
import sys
import math
import shutil
import random
import traceback
import subprocess
import uuid
import numpy as np
import pdb
import multiprocessing.pool
from multiprocessing import Pool
import multiprocessing


#A2C Dependencies
# sys.path.append("nnrunner/a2c_gvgai")
# import env
# import model
# import runner
# import tensorflow as tf
# import level_selector as ls
# import baselines.ppo2.policies as policies
# tf.logging.set_verbosity(tf.logging.FATAL)

# def calculateDijkstraMap(genes, start, solids):
#     result = []
#     for y in range(0, len(genes)):
#         result.append([])
#         for x in range(0, len(genes[y])):
#             result[y].append(-1)
#     directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
#     queue = [start]
#     result[start[1]][start[0]] = 0
#     while(len(queue) > 0):
#         current = queue.pop()
#         for dir in directions:
#             next = (current[0] + dir[0], current[1] + dir[1])
#             if next[0] < 0 or next[0] >= len(genes[0]) or next[1] < 0 or next[1] >= len(genes):
#                 continue
#             if genes[next[1]][next[0]] in solids:
#                 continue
#             if result[next[1]][next[0]] == -1 or result[current[1]][current[0]] + 1 < result[next[1]][next[0]]:
#                 result[next[1]][next[0]] = result[current[1]][current[0]] + 1
#                 queue.append(next)
#     return result

# def updateIterationResults(filePath, iteration, map):
#     file = open(filePath + "results.txt", "a")
#     cells = map.getCells()
#     numberOfNNFit = 0
#     numberOfTSFit = 0
#     for c in cells:
#         feasible = c.getFeasibleChromosomes()
#         for f in feasible:
#             if max(f._results["NN"]["win"]) == 1:
#                 numberOfNNFit += 1
#             if max(f._results["TS"]["win"]) == 1:
#                 numberOfTSFit += 1
#     file.write("Iteration " + str(iteration) + ": " + str(len(map.getCells())) + " " + str(numberOfNNFit) + " " + str(numberOfTSFit) + "\n")
#     file.close()

# def writeIteration(filePath, iteration, map):
#     os.mkdir(filePath + str(iteration) + "/")
#     map.writeMap(filePath + str(iteration) + "/")

# def deleteIteration(filePath, iteration):
#     if os.path.exists(filePath + str(iteration) + "/"):
#         shutil.rmtree(filePath + str(iteration) + "/")


sys.path.append("nnrunner/a2c_gvgai")
import env
import model
import runner
import tensorflow as tf
import level_selector as ls
import baselines.ppo2.policies as policies
tf.logging.set_verbosity(tf.logging.FATAL)


def calculateDijkstraMap(genes, start, solids):
    result = []
    for y in range(0, len(genes)):
        result.append([])
        for x in range(0, len(genes[y])):
            result[y].append(-1)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = [start]
    result[start[1]][start[0]] = 0
    while(len(queue) > 0):
        current = queue.pop()
        for dir in directions:
            next = (current[0] + dir[0], current[1] + dir[1])
            if next[0] < 0 or next[0] >= len(genes[0]) or next[1] < 0 or next[1] >= len(genes):
                continue
            if genes[next[1]][next[0]] in solids:
                continue
            if result[next[1]][next[0]] == -1 or result[current[1]][current[0]] + 1 < result[next[1]][next[0]]:
                result[next[1]][next[0]] = result[current[1]][current[0]] + 1
                queue.append(next)
    return result

def updateIterationResults(filePath, iteration, map):
    #print(filePath + str(iteration) + "/results.txt")
    with open(filePath + "/results.txt", "a") as file:
        cells = map.getCells()
        numberOfNNFit = 0
        numberOfTSFit = 0
        for c in cells:
            feasible = c.getFeasibleChromosomes()
            for f in feasible:
                if max(f._results["NN"]["win"]) == 1:
                    numberOfNNFit += 1
                if max(f._results["TS"]["win"]) == 1:
                    numberOfTSFit += 1
        file.write("Iteration " + str(iteration) + ": "     + str(len(map.getCells())) + " " + str(numberOfNNFit) + " " + str(numberOfTSFit) + "\n")

def writeIteration(filePath, iteration, map):
    #print(filePath)
    #print(filePath + str(iteration) + "/")
    os.makedirs(filePath + str(iteration) + "/")
    map.writeMap(filePath + str(iteration) + "/")

def deleteIteration(filePath, iteration):
    if os.path.exists(filePath + str(iteration) + "/"):
        shutil.rmtree(filePath + str(iteration) + "/")


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
        
        

class Zelda:
    def __init__(self, lvlFilename, resultFilename):
        self._name = "zelda"
        self._charMapping = [".", "w", "A", "g", "+", "1", "2", "3"]
        self._minValues = {"g": 1, "+": 1, "A": 1}
        self._maxValues = {"g": 1, "+": 1, "A": 1, "1": 5, "2": 5, "3": 5}
        self._avatar = 2
        self._solid = 1
        self._empty = 0
        self._enemies = [5, 6, 7]
        self._door = 3
        self._key = 4
        self._scores = {"g": 1, "+": 1, "1": 2, "2": 2, "3": 2}
        self._lvlFilename = lvlFilename
        self._resultFilename = resultFilename
        
        #A2C Variables
        # pdb.set_trace()
        self._levelSelector = ls.LevelSelector.get_selector("map-elite", os.path.basename(lvlFilename), 
                                                       os.path.dirname(os.path.abspath(lvlFilename)), max=1)
        # pdb.set_trace()
        self._gymEnv = env.make_gvgai_env("gvgai-zelda-lvl0-v0", 1, 0, level_selector=self._levelSelector)
        self._agentModel = model.Model(policy=policies.CnnPolicy, ob_space=self._gymEnv.observation_space, 
                                       ac_space=self._gymEnv.action_space, nenvs=1, nsteps=5)
        self._agentModel.load('nnrunner/a2c_gvgai/results/zelda-pcg-progressive/models/zelda100m/', 100000000)
        self._gymEnv.reset()

    def getMaxScore(self, map):
        total = 0
        for y in range(0, len(map)):
            for x in range(0, len(map[y])):
                ch = self.getIndexToChar(map[y][x])
                if ch in self._scores:
                    total += self._scores[ch]
        return total

    def getRandomValue(self, map):
        chars = self._charMapping.copy()
        for ch in self._maxValues:
            if len(self.getLocations(map, self.getCharToIndex(ch))) >= self._maxValues[ch] and ch in chars:
                chars.remove(ch)
        return self._charMapping.index(random.choice(chars))

    def getLocations(self, map, index):
        result = []
        for y in range(0, len(map)):
            for x in range(0, len(map[y])):
                if map[y][x] == index:
                    result.append((x, y))
        return result

    def getHistogram(self, map):
        histogram = [0]*len(self._charMapping)
        for y in range(0, len(map)):
            for x in range(0, len(map[y])):
                histogram[map[y][x]] += 1
        return histogram

    def getNonSatisfyMinMax(self, map):
        result = set()
        for y in range(0, len(map)):
            for x in range(0, len(map[y])):
                ch = self.getIndexToChar(map[y][x])
                number = len(self.getLocations(map, map[y][x]))
                if (ch in self._minValues and number < self._minValues[ch]) or (ch in self._maxValues and number > self._maxValues[ch]):
                    result.add(map[y][x])
        return list(result)

    def getNonConnected(self, map):
        if len(self.getLocations(map, self._avatar)) == 0:
            return len(self._charMapping)

        dijkstra = calculateDijkstraMap(map, self.getLocations(map, self._avatar)[0], [self._solid])
        numErrors = 0

        error = 0
        positions = self.getLocations(map, self._key)
        for pos in positions:
            if dijkstra[pos[1]][pos[0]] == -1:
                error += 1
        if len(positions) > 0:
            numErrors += error / len(positions)

        error = 0
        positions = self.getLocations(map, self._door)
        for pos in positions:
            if dijkstra[pos[1]][pos[0]] == -1:
                error += 1
        if len(positions) > 0:
            numErrors += error / len(positions)

        for e in self._enemies:
            error = 0
            positions = self.getLocations(map, e)
            for pos in positions:
                if dijkstra[pos[1]][pos[0]] == -1:
                    error += 1
            if len(positions) > 0:
                numErrors += error / len(positions)

        return numErrors

    def getEntropy(self, map):
        area = 1.0 * len(map) * len(map[0])
        emptyTiles = len(self.getLocations(map, self._empty))
        return emptyTiles/area
    
    def getBorder(self):
        return self._solid

    def getCharToIndex(self, char):
        return self._charMapping.index(char)

    def getIndexToChar(self, index):
        return self._charMapping[index]

    def getDimensions(self, map):
        area = 1.0 * len(map) * len(map[0])
        maxLength = 1.0 * (len(map) + len(map[0]))
        emptyTiles = len(self.getLocations(map, self._empty))
        numEnemies = 0
        for e in self._enemies:
            locs = self.getLocations(map, e)
            numEnemies += len(locs)

        if len(self.getLocations(map, self._avatar)) == 0:
            return [round(emptyTiles / area * 10 + 0.5) - 1, min(numEnemies, 10), 0, 0, 0]

        dijkstra = calculateDijkstraMap(map, self.getLocations(map, self._avatar)[0], [self._solid])
        nearestEnemy = maxLength
        for e in self._enemies:
            locs = self.getLocations(map, e)
            for p in locs:
                if dijkstra[p[1]][p[0]] >= 0 and dijkstra[p[1]][p[0]] < nearestEnemy:
                    nearestEnemy = dijkstra[p[1]][p[0]]
        nearestKey = maxLength
        locs = self.getLocations(map, self._key)
        for p in locs:
            if dijkstra[p[1]][p[0]] >= 0 and dijkstra[p[1]][p[0]] < nearestKey:
                nearestKey = dijkstra[p[1]][p[0]]
        nearestDoor = maxLength
        locs = self.getLocations(map, self._door)
        for p in locs:
            if dijkstra[p[1]][p[0]] >= 0 and dijkstra[p[1]][p[0]] < nearestDoor:
                nearestDoor = dijkstra[p[1]][p[0]]

        return [round(emptyTiles / area * 10 + 0.5) - 1, min(numEnemies, 10), round(nearestEnemy / maxLength * 10 + 0.5) 
                - 1, round(nearestKey / maxLength * 10 + 0.5) - 1, round(nearestDoor / maxLength * 10 + 0.5) - 1]
    
    def runNN(self, level, iteration=0):
        print("runNN")
        random = str(uuid.uuid1())
        fileName = self._lvlFilename.replace("level.txt",random + "_level.txt")
        # fileName = self._lvlFilename
        with open(fileName, "w") as f:
            f.write(level)

        nh, nw, nc = self._gymEnv.observation_space.shape
        obs = np.zeros((1, nh, nw, nc), dtype=np.uint8)
        model_states = self._agentModel.initial_state
        dones = [False]
        while not dones[0]:
            #Sself._gymEnv.render()
            actions, values, model_states, _ = self._agentModel.step(obs, model_states, dones)
            obs, rewards, dones, info = self._gymEnv.step(actions)

        win = 1 if (info[0]['winner'] == 'PLAYER_WINS') else 0
        score = info[0]['episode']['r']
        steps = info[0]['episode']['l']
        time = info[0]['episode']['t']
        
        os.remove(fileName)
        return [win, score, steps]

    def runTS(self, level, iteration=0):
        print("runTS")
        random = str(uuid.uuid1())
        fileName = self._lvlFilename.replace("level.txt",random + "_level.txt")
        # fileName = self._lvlFilename
        if not os.path.exists(self._lvlFilename.replace('/level.txt','')):
            os.mkdir(self._lvlFilename.replace('/level.txt',''))
            
        with open(fileName, "w") as f:
            f.write(level)

        e = ["java", "-jar", "tsrunner/tsrunner.jar", "tsrunner/examples/gridphysics/" + self._name + ".txt", 
             self._lvlFilename, self._resultFilename, str(iteration)]
        p = subprocess.run(e)
        
        with open(self._resultFilename) as f:
            parts = f.readlines()[0].split(",")
        os.remove(fileName)
        os.remove(self._resultFilename)
        return [float(parts[0]), float(parts[1]), float(parts[2])];

    def __del__(self):
        try:
            self._gymEnv.close()
        except:
            pass
