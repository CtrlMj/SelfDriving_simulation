from imgaug import augmenters as ia  
import numpy as np
import cv2
import matplotlib.image as mpimg

def preprocess(image):
  image = image[60:137, :, :]                                                   #first we crop the image. Like color of the sky is not important for us.
  image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)                                #We change the color channel since Nvidia model that will be implemented will work best with this channel
  image = cv2.GaussianBlur(image, (3, 3), 0)                                    #reduce the noise and smooth out the image
  image = cv2.resize(image, (200, 66))                                          #this is not necassary but Nvidia model has been trained using this size of images so will probably perform better on this size
  image = image / 255
  return image


def zoom(image):
  zoom = ia.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image

def pan(image):
  pan = ia.Affine(translate_percent={"x": (-0.1, 0.1), "y":(-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

def darken(image):
  darken = ia.Multiply((0.2, 1.2))
  image = darken.augment_image(image)
  return image

def flip(image, steering):
  image = cv2.flip(image, 1)
  steering = -1*steering
  return image, steering



def augmentor(image, steering):
  image = mpimg.imread(os.path.join('SelfDriving', 'IMG', image))
  if np.random.rand() > 0.5:
    image = zoom(image)
  if np.random.rand() > 0.5:
    image = pan(image)
  if np.random.rand() > 0.5:
    image = darken(image)
  if np.random.rand() > 0.5:
    image, steering = flip(image, steering)
  return image, steering


def batch_generator(images, steerings, batch_size, train):
  while True:
    imagebatch = []
    steerbatch = []
    indices = np.random.randint(len(images), size=batch_size)
    if train:
      maps = list(map(augmentor, images[indices], steerings[indices]))
      imagebatch, steerbatch = zip(*maps)
    else:
      imagebatch = list(map(mpimg.imread, 'SelfDriving/' + 'IMG/' + images[indices]))
      steerbatch = steerings[indices]
      
    imagebatch = list(map(preprocess, imagebatch))
    yield (np.asarray(imagebatch), np.asarray(steerbatch))
