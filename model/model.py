import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam

def nvidia(lr=0.005, dropouts = [0, 0, 0.3]):
  model = Sequential()
  #subsample defines the strides of the kernel by which it traverses
  model.add(Convolution2D(24, (5, 5), subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
  #we use elu instead of relu so that nodes dont die
  model.add(Convolution2D(36, (5, 5), subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(48, (5, 5), subsample=(2, 2), activation='elu'))
  #by this point the input has become pretty small so no need for stride change Also size of the kernel should change
  model.add(Convolution2D(64, (3, 3), activation='elu'))                        
  model.add(Convolution2D(64, (3, 3), activation='elu'))
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dropout(dropouts[0]))
  model.add(Dense(50,  activation='elu'))
  model.add(Dropout(dropouts[1]))
  model.add(Dense(10,  activation='elu'))
  model.add(Dropout(dropouts[2]))
  model.add(Dense(1))
  model.compile(optimizer=Adam(0.005), loss='mse', metrics=['accuracy'])
  return model
