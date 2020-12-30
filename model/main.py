from Preprocessing import preprocess
from Preprocessing import zoom, pan, darken, flip, augmentor
from Preprocessing import batch_generator
from model import nvidia
import sys
sys.path.append('SelfDriving_simulation/data')


EPOCHS = 10
TRAIN_DATA = batch_generator(X_train, y_train, batch_size=100, train=1)
VALIDATION_DATA = batch_generator(X_test, y_test, batch_size=100, train=0)


model = nvidia()
print(f"Summary of the model:\n{model1.summary()}")

print("Training starts")
history = model.fit_generator(TRAIN_DATA, steps_per_epoch=300, epochs=10,
                              validation_data = VALIDATION_DATA, 
                              validation_steps=200, verbose=1, shuffle=1)  


print("Final loss plots for training and validation:")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('Epoch')
plt.show()
