import os, ntpath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv("SelfDriving/log.csv", names = ["center", "left", "right", "steering", "throttle", "reverse", "speed"])
data['center'] = data['center'].apply(lambda x: ntpath.basename(x))

def render_raw(path="SelfDriving/log.csv"):
  data = pd.read_csv(path, names = ["center", "left", "right", "steering", "throttle", "reverse", "speed"])
  data['center'] = data['center'].apply(lambda x: ntpath.basename(x))
  
  to_be_removed = []                                                           
  for i in range(25):
    lst = []                                                                     
    for j in range(len(data['steering'])):
        if data['steering'][j] <= bins[i+1] and data['steering'][j] >= bins[i]:
        lst.append(j)
    lst = shuffle(lst)                                                            
    to_be_removed.extend(lst[threshold:])
  data.drop(data.index[to_be_removed], inplace=True)
  imagepath = np.array(data['center'])
  steer = np.array(data['steering'])
  X_train, X_test, y_train, y_test = train_test_split(imagepath, steer, test_size = 0.2, random_state = 1)
  return X_train, X_test, y_train, y_test
