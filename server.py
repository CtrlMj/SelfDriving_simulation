import socketio
import numpy as np
from flask import Flask
import eventlet
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

LOAD_PATH = './load_path'
sio = socketio.Server()
app = Flask(__name__)

speed_limit = 10
def preprocess(image):
  image = image[60:137, :, :]                                                   #first we cut the crap from our image. Like color of the sky is not important for us.
  image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)                                #We change the color channel since Nvidia model that will be implemented will work best with this channel
  image = cv2.GaussianBlur(image, (3, 3), 0)                                    #reduce the noise and smooth out the image
  image = cv2.resize(image, (200, 66))                                          #this is not necassary but Nvidia model has been trained using this size of images so will probably perform better on this size
  image = image / 255
  return image


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.array(image)
    image = preprocess(image)
    image = np.array([image])
    steer = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    send_steer(steer, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_steer(0, 1)

def send_steer(angle, throttle):
    sio.emit('steer', data={
        'steering_angle': angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model(LOAD_PATH)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
