import os
import gym
import gym_donkeycar
import numpy as np
import cv2
from pid import TwiddlingPID
import csv
import math

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
# This assumes DonkeySim simulator is extracted at root of thi repo.
exe_path = ".\\DonkeySimWin\\donkey_sim.exe"
port = 9091

conf = { 
         #"exe_path" : exe_path,
         "port" : port,
         "body_style": "car01", # cybertruck if that's what you like
         "body_rgb": (255, 0, 255),
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

waypoints = []
waypoint_index = 0

# PLAY
obs = env.reset()

with open('path_log.csv', newline='') as csvfile:
    pathreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in pathreader:
        waypoints.append([float(row[2]),float(row[0])])

def nearest_waypoint_index(pos,search_offset):
    dist2 = 1000000
    nearest_index  = 0
    for i in range(20):
        w = waypoints[(search_offset-10+i)%len(waypoints)]
        d2 = (w[0]-pos[2])*(w[0]-pos[2])+(w[1]-pos[0])*(w[1]-pos[0])
        if d2<dist2:
            dist2 =d2
            nearest_index=(search_offset-10+i)%len(waypoints)
    return nearest_index

for t in range(2000):
    action = np.array([steer_value, 0.15+0.05*math.sin(t*0.05)]) # Osccilating speed
    # execute the action
    obs, reward, done, info = env.step(action)

    # LATERAL CONTROL
    # This is a crude control based on recorded waypoints
    pos = info['pos']

    waypoint_index = nearest_waypoint_index(pos,waypoint_index)
    lookup_point = (waypoint_index+4)%len(waypoints)

    target_angle = math.atan2(waypoints[lookup_point][1]-pos[0],waypoints[lookup_point][0]-pos[2])
    yaw = info['car'][2]*math.pi/180.0

    diff = yaw-target_angle
    if(diff<-math.pi):
        diff=diff+2*math.pi
    if(diff>math.pi):
        diff=diff-2*math.pi
    
    pid.update_error(diff)
    steer_value = pid.total_error()
  
    if steer_value > 1.0:
      steer_value = 1.0
    if steer_value < -1.0:
      steer_value = -1.0

# Exit the scene
env.close()