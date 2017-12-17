import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

path = "..//dataset//test"
files = os.listdir(path)

Xs = []
ys = []

for filename in files:
    img = plt.imread(os.path.join(path, filename))

    X = cv2.resize(img, (32, 32)) 
    y = int(filename[:filename.find("_")])
    
    Xs.append(X)
    ys.append(y)

Xs = np.array(Xs)
print(Xs.shape)
print(ys)
