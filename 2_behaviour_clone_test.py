import os
import gym
import gym_donkeycar
import numpy as np
import cv2

from keras.models import load_model
import h5py

from model import preprocess

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
# This assumes DonkeySim simulator is extracted at root of thi repo.
exe_path = ".\\DonkeySimWin\\donkey_sim.exe"
port = 9091

conf = { 
         "exe_path" : exe_path,
         "port" : port,
         "body_style": "car01", # cybertruck if that's what you like
         "body_rgb": (128, 128, 128),
         "car_name": "",
         "font_size": 100 
       }

env = gym.make("donkey-minimonaco-track-v0", conf=conf)

# check that model Keras version is same as local Keras version
#f = h5py.File('model.h5', mode='r')

model = load_model('model.h5')

steering_angle = 0

# PLAY
obs = env.reset()
for t in range(250):
  action = np.array([steering_angle, 0.15]) # drive straight with small speed
  # execute the action
  obs, reward, done, info = env.step(action)

  image_array = preprocess(np.asarray(obs))
  #if(t==0):
  #  cv2.imwrite("sample.jpg",image_array)
  steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
  print(steering_angle)

# Exit the scene
env.close()