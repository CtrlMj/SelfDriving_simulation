from Preprocessing import preprocess
from Preprocessing import zoom, pan, darken, flip, augmentor
from Preprocessing import batch_generator
from model import nvidia
import sys
sys.path.append('SelfDriving_simulation/data')
from SelfDriving_simulation.data import render_raw

EPOCHS = 10
lr = 0.005
data_path = "SelfDriving/log.csv"
test_size = 0.2
n_bins = 25
unbalance_thresh = 300
batch_size = 100
filepath="./cpath"


if __name__ == "__main__":
  X_train, y_train, X_test, y_test = render_raw(path=data_path, testsize=test_size, n_bins=n_bins, threshold=unbalance_thresh)
  TRAIN_DATA = batch_generator(X_train, y_train, batch_size=batch_size, train=1)
  VALIDATION_DATA = batch_generator(X_test, y_test, batch_size=batch_size, train=0)


  model = nvidia(lr, [0, 0, 0.3])
  print(f"Summary of the model:\n{model.summary()}")
  
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  print("Training starts")
  history = model.fit_generator(TRAIN_DATA, steps_per_epoch=300, epochs=EPOCHS,
                                validation_data = VALIDATION_DATA, 
                                validation_steps=200, verbose=1, callbacks=callbacks_list, shuffle=1)  


  print("Final loss plots for training and validation:")
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.legend(['training', 'validation'])
  plt.title('loss')
  plt.xlabel('Epoch')
  plt.show()


