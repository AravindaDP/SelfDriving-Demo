import os
import gym
import gym_donkeycar
import numpy as np

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
# This assumes DonkeySim simulator is extracted at root of thi repo.
exe_path = f".\\DonkeySimWin\\donkey_sim.exe"
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

# PLAY
obs = env.reset()
for t in range(100):
  action = np.array([0.0, 0.3]) # drive straight with small speed
  # execute the action
  obs, reward, done, info = env.step(action)

# Exit the scene
env.close()