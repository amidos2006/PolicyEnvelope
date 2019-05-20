import matplotlib.image as mpimg
import cv2
from scipy import misc
import numpy as np

def getImage(tiles, tilesDic):
    lines = tiles.split("\n")
    for i in range(0, len(lines)):
        if len(lines[i].strip()) == 0:
            del lines[i]
            i-=1
    width = len(lines[0].strip())
    height = len(lines)
    image = np.zeros((height * 24, width * 24, 3))
    for yTile in range(0, len(lines)):
        for xTile in range(0, len(lines[yTile])):
            t = lines[yTile][xTile]
            if t in tilesDic:
                ct = tilesDic[t]
                ct = cv2.resize(ct, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
                image[yTile*24:(yTile + 1)*24,xTile*24:(xTile + 1)*24,:] = ct[:,:,0:3]
    return image

tilesDic = {}
tilesDic['.'] = mpimg.imread('tsrunner/sprites/oryx/floor3_0.png')
tilesDic['A'] = mpimg.imread('tsrunner/sprites/oryx/swordman1_0.png')
tilesDic['g'] = mpimg.imread('tsrunner/sprites/oryx/doorclosed1.png')
tilesDic['+'] = mpimg.imread('tsrunner/sprites/oryx/key2.png')
tilesDic['w'] = mpimg.imread('tsrunner/sprites/oryx/wall3_0.png')
tilesDic['1'] = mpimg.imread('tsrunner/sprites/oryx/bat1.png')
tilesDic['2'] = mpimg.imread('tsrunner/sprites/oryx/spider2.png')
tilesDic['3'] = mpimg.imread('tsrunner/sprites/oryx/scorpion1.png')
# print(tilesDic["."].shape)
# tiles = "wwwwwwwwwwwww\nwA.......w..w\nw..w........w\nw...w...w.+ww\nwww.w2..wwwww\nw.......w.g.w\nw.2.........w\nw.....2.....w\nwwwwwwwwwwwww\n"

# test = getImage(tiles, tilesDic)
# misc.imsave('test.png', test)
