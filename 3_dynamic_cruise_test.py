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

degAngDown = 15
degAngDelta = -4

conf = { 
         "exe_path" : exe_path,
         "port" : port,
         "body_style": "car01", # cybertruck if that's what you like
         "body_rgb": (128, 128, 128),
         "car_name": "",
         "font_size": 100,
         "degPerSweepInc" : "4",
         "degAngDown" : degAngDown,
         "degAngDelta" : degAngDelta,
         "numSweepsLevels" : "4",
         "maxRange" : "50.0",
         "noise" : "0.4",
         "offset_x" : "0.0",
         "offset_y" : "0.5",
         "offset_z" : "0.0",
         "rot_x" : "0.0"
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
_throttle = 0.1
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

def psuedo_radar(lidar,nb_layers):
    """
    Calculates nearest obstacle point by choosing lidar points on forward direction and
    then removing points that lies on ground (based on approximate height of lidar to be 1m)
    """
    # TODO: enhance nearest obstacle with speed information based on previous nearest point information.
    pts_per_layer = len(lidar)//nb_layers
    x_min = 100
    y_min = 100
    l_min = 100
    for i in range(nb_layers):
        layer_angle_deg = degAngDown+(i*degAngDelta)
        layer_points = lidar[i*pts_per_layer:i*pts_per_layer+pts_per_layer//24+1]
        layer_points2 =lidar[(i+1)*pts_per_layer-pts_per_layer//24:(i+1)*pts_per_layer]
        for p in range(len(layer_points)):
            point = layer_points[p]
            if point>0:
                h  = point * math.sin(layer_angle_deg*math.pi/180.0)
                l  = point * math.cos(layer_angle_deg*math.pi/180.0)
                if(h<1 and l<l_min):
                    y = l * math.sin(p*2*math.pi/pts_per_layer)
                    x = l * math.cos(p*2*math.pi/pts_per_layer)
                    x_min = x
                    y_min = y
                    l_min = l
        for p in range(len(layer_points2)):
            point = layer_points2[-(p+1)]
            if point>0:
                h  = point * math.sin(layer_angle_deg*math.pi/180.0)
                l  = point * math.cos(layer_angle_deg*math.pi/180.0)
                if(h<1 and l<l_min):
                    y = -l * math.sin((p+1)*2*math.pi/pts_per_layer)
                    x = l * math.cos((p+1)*2*math.pi/pts_per_layer)
                    x_min = x
                    y_min = y
                    l_min = l
    return (x_min,y_min)

for t in range(100):
    action = np.array([steer_value, 0.0]) # drive straight with small speed
    # execute the action
    obs, reward, done, info = env.step(action)

for t in range(2000):
    action = np.array([steer_value, _throttle]) # drive straight with small speed
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

    # SPEED CONTROL
    # TODO: Use a dedicated longitudinal control pid to smoothly control speed.
    lidar = info['lidar']
    obs_x, obs_y  =  psuedo_radar(lidar,4)
    if abs(obs_y)>2.5:
        _throttle = 0.2
    else:
        _throttle = obs_x*0.0075
        if(_throttle)>0.25:
            _throttle = 0.25
        if(_throttle)<0.1:
            _throttle = 0.1
    print(_throttle)

# Exit the scene
env.close()