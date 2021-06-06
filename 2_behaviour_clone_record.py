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

def record(offset,count,cte_offset,correction,csvwriter):
  global steer_value
  for t in range(count):
    action = np.array([steer_value, 0.1]) # drive straight with small speed
    # execute the action
    obs, reward, done, info = env.step(action)

    rgb_image = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    cv2.imwrite("data/MiniMonaco/IMG/"+str(t+offset)+".jpg",rgb_image)
    
    cte = info['cte']-cte_offset
    pid.update_error(cte)
    steer_value = pid.total_error()
    record_value = steer_value+correction
  
    if steer_value > 1.0:
      steer_value = 1.0
    if steer_value < -1.0:
      steer_value = -1.0

    if record_value > 1.0:
      record_value = 1.0
    if record_value < -1.0:
      record_value = -1.0

    csvwriter.writerow([str(t+offset)+".jpg",str(record_value)])

# PLAY
obs = env.reset()

with open('data/MiniMonaco/driving_log.csv', 'w', newline='') as csvfile:
  csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
  
  #Record middle lane driving
  record(0,750,1.95,0,csvwriter)

  # slowly offset to left
  for t in range(100):
    action = np.array([steer_value, 0.1]) # drive straight with small speed
    # execute the action
    obs, reward, done, info = env.step(action)
    
    cte = info['cte']-1.95+(0.0025*t)
    pid.update_error(cte)
    steer_value = pid.total_error()
  
    if steer_value > 1.0:
      steer_value = 1.0
    if steer_value < -1.0:
      steer_value = -1.0

  #Record sligtly left in the lane with slight positive adjustment to steering values
  record(750,500,1.7,0.15,csvwriter)

  # slowly offset to right
  for t in range(100):
    action = np.array([steer_value, 0.1]) # drive straight with small speed
    # execute the action
    obs, reward, done, info = env.step(action)
    
    cte = info['cte']-1.7-(0.005*t)
    pid.update_error(cte)
    steer_value = pid.total_error()
  
    if steer_value > 1.0:
      steer_value = 1.0
    if steer_value < -1.0:
      steer_value = -1.0

  #Record sligtly right in the lane with slight negative adjustment to steering values
  record(1250,500,2.2,-0.15,csvwriter)

# Exit the scene
env.close()