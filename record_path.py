import os
import gym
import gym_donkeycar
import numpy as np
import cv2
from pid import TwiddlingPID
import csv

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

# SET UP PID

init_dp = [0, 0, 0]
pid = TwiddlingPID(init_dp,4.0,2000)
# TODO: Initialize the pid variable.
# Need to tune these values for better pid handling
init_Kp = 0.7
init_Kd = 2.0
init_Ki = 0.0001
pid.init(init_Kp, init_Ki, init_Kd)
print("Kp: ", pid.Kp, " Ki: ", pid.Ki, " Kd: ", pid.Kd)
_throttle = 0.0
_max_speed = 0.0

steer_value = 0

# PLAY
obs = env.reset()

with open('path_log.csv', 'w', newline='') as csvfile:
  csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
  
  for t in range(2200):
    action = np.array([steer_value, 0.1]) # drive straight with small speed
    # execute the action
    obs, reward, done, info = env.step(action)
    
    cte = info['cte']-1.95
    pid.update_error(cte)
    steer_value = pid.total_error()
  
    if steer_value > 1.0:
      steer_value = 1.0
    if steer_value < -1.0:
      steer_value = -1.0

    if t%5==0:
        csvwriter.writerow([info['pos'][0],info['pos'][1],info['pos'][2],info['car'][0],info['car'][1],info['carx'][2]])

# Exit the scene
env.close()