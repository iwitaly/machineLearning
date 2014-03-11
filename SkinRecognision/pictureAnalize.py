import os,sys
from PIL import Image
import numpy as np
import main

predictor = main.PicturePredictor()

im = Image.open("dasha1.jpg")
pixels = im.load()
width, height = im.size

matr = np.array(im).astype(int)
predictionMatrix = []

for row in matr:
    predictor.loadTestSetFromArray(row)
    predictionMatrix.append(predictor.predictData())

for x in range(width):
    for y in range(height):
        if predictionMatrix[y][x] == 1:
            pixels[x, y] = (127, 242, 0)
im.save('out.jpg')