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
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

sys.path.append("nnrunner/a2c_gvgai")
import env
import model
import runner
import tensorflow as tf
import level_selector as ls
import baselines.ppo2.policies as policies
from gym.envs.classic_control import rendering
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

def connectedComponent(genes, target, replace):
    
    count = 0
    
    def addBorder(lvl):
        expand = np.array([1]*lvl.shape[0])[:,None]
        lvl = np.concatenate((expand, lvl, expand), axis=1)
        lvl = np.vstack(([1]*(lvl.shape[1]), lvl, [1]*(lvl.shape[1])))
        return lvl
    
    def rp(lvl, Q, node, target, replace):
        if lvl[node[0],node[1]] == target:
            lvl[node[0],node[1]] = replace
            Q.append(node)

    lvl = np.array(genes.copy())
    lvl = addBorder(lvl)
    
    if target == replace or target not in lvl:
        return 0
    
    Q = []
    rp(lvl, Q, list(zip(*np.where(lvl==target)))[0], target, replace)
    while len(Q) != 0:
        x,y = Q[0]
        Q.pop(0)
        nb = []
        # up left corner
        if x == 0 and y == 0 :
            nb = [[x,y+1], [x+1,y], [x+1,y+1]]
        # up right corner
        elif x == 0 and y == lvl.shape[1]-1 :
            nb = [[x,y-1], [x+1,y-1], [x+1,y]]
        # bottom left corner
        elif x == lvl.shape[0]-1 and y == 0:
            nb = [[x-1,y], [x-1,y+1], [x,y+1]]
        # bottom right corner
        elif x == lvl.shape[0]-1 and y == lvl.shape[1]-1 :
            nb = [[x-1,y-1], [x,y-1], [x-1,y]]
        # up 
        elif x == 0 :
            nb = [[x,y-1], [x+1,y-1], [x+1,y], [x+1,y+1], [x,y+1]]
        # bottom
        elif x == lvl.shape[0]-1 :
            nb = [[x-1,y-1], [x,y-1], [x-1,y], [x-1,y+1], [x,y+1]]
        # left
        elif y == 0 :
            nb = [[x-1,y], [x-1,y+1], [x,y+1], [x+1,y+1], [x+1,y]]
        # right
        elif y == lvl.shape[1]-1 :
            nb = [[x-1,y-1], [x-1,y], [x,y-1], [x+1,y-1], [x+1,y]]
        else:
            nb = [[x-1,y-1], [x-1,y], [x-1,y+1], [x,y-1], [x,y+1], [x+1,y-1], [x+1,y], [x+1,y+1]]
        for i in nb:
            rp(lvl, Q, i, target, replace)
        if len(Q) == 0:
            count += 1
            targets = list(zip(*np.where(lvl==target)))
            if targets:
                rp(lvl, Q, targets[0], target, replace)
                
            
    return count
    

def updateIterationResults(filePath, iteration, map):
    #print(filePath + str(iteration) + "/results.txt")
    if not os.path.exists(filePath):
        os.makedirs(filePath)
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


        
        

class Zelda:
    def __init__(self, lvlFilename, resultFilename, numOfTests):
        self._name = "zelda"
        self._charMapping = [".", "w", "A", "g", "+", "1", "2", "3"]
        # self._charMapping = [".", "w", "A", "g", "+"]
        self._minValues = {"g": 1, "+": 1, "A": 1}
        self._maxValues = {"g": 1, "+": 1, "A": 1, "1": 5, "2": 5, "3": 5}
        # self._maxValues = {"g": 1, "+": 1, "A": 1}
        self._avatar = 2
        self._solid = 1
        self._empty = 0
        self._enemies = [5, 6, 7]
        self._door = 3
        self._key = 4
        self._scores = {"g": 1, "+": 1, "1": 2, "2": 2, "3": 2}
        # self._scores = {"g": 1, "+": 1}
        self._lvlFilename = lvlFilename
        self._resultFilename = resultFilename
        
        #A2C Variables
        self._levelSelector = ls.LevelSelector.get_selector("map-elite", os.path.basename(lvlFilename), 
                                                       os.path.dirname(os.path.abspath(lvlFilename)), max=1)
        # self._levelSelector = ls.MapEliteSelector(os.path.dirname(os.path.abspath(lvlFilename)), os.path.basename(lvlFilename)
        self._gymEnv = env.make_gvgai_env("gvgai-zelda-lvl0-v0", numOfTests, 0, level_selector=self._levelSelector)
        self._agentModel = model.Model(policy=policies.CnnPolicy, ob_space=self._gymEnv.observation_space, 
                                       ac_space=self._gymEnv.action_space, nenvs=numOfTests, nsteps=5)
        self._agentModel.load('nnrunner/a2c_gvgai/results/zelda-pcg-progressive/models/zelda100m/', 100000000)
        # self._gymEnv.reset()
        self._reset = False

        # self.runNN("wwwwwwwwwwwww\nwA.......w..w\nw..w........w\nw...w...w.+ww\nwww.w2..wwwww\nw.......w.g.w\nw.2.........w\nw.....2.....w\nwwwwwwwwwwwww", numOfTests)

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
        for k,v in self._minValues.items():
            if not any(self.getCharToIndex(k) in s for s in map):
                result.add(self.getCharToIndex(k))
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
        emptyTiles = 1.0 * len(self.getLocations(map, self._empty))
        return emptyTiles/area
    
    def getBorder(self):
        return self._solid

    def getCharToIndex(self, char):
        return self._charMapping.index(char)

    def getIndexToChar(self, index):
        return self._charMapping[index]

    def getRecords(self, map):
        area = 1.0 * len(map) * len(map[0])
        maxLength = 1.0 * (len(map) + len(map[0]))
        emptyTiles = len(self.getLocations(map, self._empty))
        numEnemies = 0
        for e in self._enemies:
            locs = self.getLocations(map, e)
            numEnemies += len(locs)
        
        paths = calculateDijkstraMap(map, self.getLocations(map, self._empty)[np.random.randint(emptyTiles)], [self._solid])
        maxPaths = calculateDijkstraMap(map, np.unravel_index(np.argmax(paths), np.array(paths).shape)[::-1], [self._solid])
        longestPath = np.max(maxPaths)
        numConnectedComponent = connectedComponent(map, self._solid, -1)

        if len(self.getLocations(map, self._avatar)) == 0:
            return [emptyTiles , numEnemies, 0, 0, 0, longestPath, numConnectedComponent]

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
        
        
        return [emptyTiles, numEnemies, nearestEnemy, nearestKey, nearestDoor, longestPath, numConnectedComponent]
    
    def getDimensions(self, map):
        area = 1.0 * len(map) * len(map[0])
        maxLength = 1.0 * (len(map) + len(map[0]))
        raw = self.getRecords(map)
        
        if len(self.getLocations(map, self._avatar)) == 0:
            return [round(raw[0] / area * 10 + 0.5) - 1, min(raw[1], 10), 0, 0, 0, raw[5]]

        return [round(raw[0] / area * 10 + 0.5) - 1, min(raw[1], 10), round(raw[2] / maxLength * 10 + 0.5) 
                - 1, round(raw[3] / maxLength * 10 + 0.5) - 1, round(raw[4] / maxLength * 10 + 0.5) - 1, raw[5]]
        
    
    def runNN(self, level, numOfTests, iteration=0):
        print("runNN")
        viewer = rendering.SimpleImageViewer()
        # fileName = self._lvlFilename.replace("level.txt",random + "_level.txt")
        fileName = self._lvlFilename
        print(level, fileName)
        with open(fileName, "w") as f:
            f.write(level)
        # if not self._reset:
        #     print("reset")
        #     self._gymEnv.reset()
        #     self._reset = True
        self._gymEnv.reset()
        nh, nw, nc = self._gymEnv.observation_space.shape
        obs = np.zeros((numOfTests, nh, nw, nc), dtype=np.uint8)
        model_states = self._agentModel.initial_state
        done = np.array([False] * numOfTests)
        dones = [False] * numOfTests
        infos = [False] * numOfTests
        frames = np.array([np.array(None)] * numOfTests)
        while not all(done):
            # print(type(self._gymEnv.envs[0]))            

            actions, values, model_states, _ = self._agentModel.step(obs, model_states, dones)
            obs, rewards, dones, info = self._gymEnv.step(actions)
            # self._gymEnv.render()
            # self._gymEnv.get_images()
            done[np.where(dones!=False)] = True
            for i in np.where(dones!=False)[0].tolist():
                if not infos[i]:
                    infos[i] = info[i]

            imgs = self._gymEnv.get_images()
            for i,img in enumerate(imgs):
                # viewer.imshow(img)
                if not done[i]:
                    if np.any(frames[i]) == None:
                        frames[i] = img[None,:,:,:]
                    else:
                        frames[i] = np.concatenate((frames[i], img[None,:,:,:]))
        # print(infos)
        
        win = [1 if (i['winner'] == 'PLAYER_WINS') else 0 for i in infos]
        score = [i['episode']['r'] for i in infos]
        steps = [i['episode']['l'] for i in infos]
        time = [i['episode']['t'] for i in infos]

        os.remove(fileName)
        print(win)
        print(score)
        print(steps)
        viewer.close()
        # frames = frames[np.array(win)==1]
        
        # for i in frames:
        # print(frames[0])
        plt.ion()
        for i,j in enumerate(frames):
            plt.figure()
            plt.title(level.split("\n")[1])
            plt.imshow(j[0])
        
        return np.array([win, score, steps]), frames

    def runTS(self, level, numOfTests,iteration=0):
        print("runTS")
        random = str(uuid.uuid4())
        # fileName = self._lvlFilename.replace("level.txt",random + "_level.txt")
        levelFileName = self._lvlFilename
        resultFileName = self._resultFilename
        parts = [False] * numOfTests
        ps = []
        
        if not os.path.exists(self._lvlFilename.replace('/level.txt','')):
            os.mkdir(self._lvlFilename.replace('/level.txt',''))
            
        levelfiles = [levelFileName.replace("level.txt",str(i) + "_level.txt") for i in range(numOfTests)] 
        resultfiles = [resultFileName.replace("result.txt",str(i) + "_result.txt") for i in range(numOfTests)]
        
        for i in range(numOfTests):
            with open(levelfiles[i], "w") as f:
                f.write(level)
        
        for i in range(numOfTests):

            e = ["java", "-jar", "tsrunner.jar", "examples/gridphysics/" + self._name + "_" + str(i) +".txt", 
                 "../" + levelfiles[i], "../"+ resultfiles[i], str(i)]
            p = subprocess.Popen(e,cwd="tsrunner/", stderr= subprocess.DEVNULL, stdout = subprocess.DEVNULL)
            
            ps.append(p)
            
        for p in ps:
            p.wait()
            
        for i in range(numOfTests):
            with open(resultfiles[i]) as f:
                parts[i] = f.readlines()[0].split(",")
                
            os.remove(levelfiles[i])
            os.remove(resultfiles[i])
        return np.array([[float(i[0]), float(i[1]), float(i[2])] for i in parts])

    def __del__(self):
        try:
            self._gymEnv.close()
        except:
            pass
