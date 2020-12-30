import os, ntpath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def render_raw(path="SelfDriving/log.csv", testsize=0.2, n_bins=25, threshold=300):
  data = pd.read_csv(path, names = ["center", "left", "right", "steering", "throttle", "reverse", "speed"])
  data['center'] = data['center'].apply(lambda x: ntpath.basename(x))
  hist, bins = np.histogram(data['steering'], n_buckets)
  
  to_be_removed = []                                                           
  for i in range(n_bins):
    lst = []                                                                     
    for j in range(len(data['steering'])):
        if data['steering'][j] <= bins[i+1] and data['steering'][j] >= bins[i]:
        lst.append(j)
    lst = shuffle(lst)                                                            
    to_be_removed.extend(lst[threshold:])
  data.drop(data.index[to_be_removed], inplace=True)
  
  imagepath = np.array(data['center'])
  steer = np.array(data['steering'])
  X_train, X_test, y_train, y_test = train_test_split(imagepath, steer, test_size = testsize, random_state = 1)
  return X_train, X_test, y_train, y_test
