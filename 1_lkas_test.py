import os
import gym
import gym_donkeycar
import numpy as np
import cv2
from image_gen import process_image
from tracker import LineTracker
from pid import TwiddlingPID

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
# This assumes DonkeySim simulator is extracted at root of thi repo.
exe_path = ".\\DonkeySimWin\\donkey_sim.exe"
port = 9091

conf = { 
         "exe_path" : exe_path,
         "port" : port,
         "body_style": "car01",
         "body_rgb": (128, 128, 128),
         "car_name": "",
         "font_size": 100 
       }

env = gym.make("donkey-minimonaco-track-v0", conf=conf)


# SET UP LANE TRACKER
thresholds = {}
thresholds['l_thresh']=(230, 255)
thresholds['b_thresh']=(170, 255)
thresholds['grad_thresh']=(10, 255)
thresholds['dir_thresh']=(np.pi*0.10, np.pi*0.40)

src = np.float32([[0,111],[49,69],[110,69],[159,111]])
dst = np.float32([[30,115],[30,5],[129,5],[129,115]])

# TODO: It may need to adjust these parameters to prevent loosing track in tight curves.
tracker_params = {}
tracker_params['window_width']=12
tracker_params['window_height']=20
tracker_params['margin']=10
tracker_params['ym_per_pix']=10/160
tracker_params['xm_per_pix']=5/(dst[2][0]-dst[1][0])
tracker_params['smooth_factor']=5

# Set up the overall class to do all the tracking
curve_centers = LineTracker(window_width = tracker_params['window_width'], window_height = tracker_params['window_height'], margin = tracker_params['margin'], ym = tracker_params['ym_per_pix'], xm = tracker_params['xm_per_pix'], smooth_factor=tracker_params['smooth_factor'])


# SET UP PID

init_dp = [0, 0, 0]
pid = TwiddlingPID(init_dp,4.0,2000)
# TODO: Initialize the pid variable.
# Need to tune these values for better pid handling
init_Kp = 0.04
init_Kd = 5.0
init_Ki = 0.0001
pid.init(init_Kp, init_Ki, init_Kd)
print("Kp: ", pid.Kp, " Ki: ", pid.Ki, " Kd: ", pid.Kd)
_throttle = 0.0
_max_speed = 0.0

steer_value = 0

# PLAY
obs = env.reset()
# TODO: Increase this range for longer simulation (Or use an infinite loop with break condition)
for t in range(200):
  action = np.array([steer_value, 0.2]) # drive with small speed
  # execute the action
  obs, reward, done, info = env.step(action)
  
  rgb_image = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

  # TODO: Uncomment these lines to capture some sample images
  #if(t < 10):
  #  cv2.imwrite("test_images/"+str(t)+".jpg",rgb_image);

  lane_image,cte,curvrad = process_image(rgb_image, src, dst, thresholds, curve_centers)

  #print("CTE: " + str(round(info['cte']-2,2)) + " Calculated: " + str(round(cte,2)))

  # TODO: Uncomment these lines for overlay images
  #if(t < 200):
  #  cv2.imwrite("output_images/"+str(t)+'_output.jpg',lane_image)

  #Calculate steer value
  # small hack to handle when tracker loose lane tracking
  if (cte == 0):
    steer_value = steer_value*0.9
  else:
    pid.update_error(cte)
    steer_value = pid.total_error()
  
  if steer_value > 1.0:
    steer_value = 1.0
  if steer_value < -1.0:
    steer_value = -1.0

# Exit the scene
env.close()