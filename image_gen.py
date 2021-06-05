import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from pipeline import pipeline 
from tracker import LineTracker

# ## Apply a perspective transform to rectify binary image to create a "birds-eye view"
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def map_lane(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

    return warped


# This expects RGB images
def process_image(img, src, dst, thresholds, tracker):
    result = pipeline(img, thresholds['l_thresh'], thresholds['b_thresh'], thresholds['grad_thresh'], thresholds['dir_thresh'])
    warped = warper(result,src,dst)
    
    left_line, right_line = tracker.find_lines(warped)
    
    road_img = tracker.get_road_img(warped)   
   
    road_warped = map_lane(road_img,src,dst)

    result = cv2.addWeighted(img,1.0,road_warped,0.5,0.0)

    curverad = 0
    cte = 0

    if left_line.detected and right_line.detected:
        ym_per_pix = tracker.ym_per_pix # meters per pixel in y dimension
        xm_per_pix = tracker.xm_per_pix # meters per pixel in x dimension

        curve_fit_cr = np.polyfit(np.array(left_line.yvals,np.float32)*ym_per_pix,np.array(left_line.bestx+right_line.bestx,np.float32)*xm_per_pix/2.0,2)
        curverad = ((1 + (2*curve_fit_cr[0]*left_line.yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])

        # calculate the offset of the car on the road
        center_diff = (left_line.line_base_pos + right_line.line_base_pos)/2
        cte = 0-center_diff
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'

        # draw the text showing curvature, offset, and speed
        cv2.putText(result, str(int(curverad))+'(m)',(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        cv2.putText(result, str(abs(round(center_diff,2)))+'m '+side_pos,(0,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)

        if left_line.line_base_pos > - 0.9 or right_line.line_base_pos < 0.9: #Approx half of average width of a car
            cv2.putText(result,'Lane Departure',(0,115),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
            # Force detecting new lane positions
            left_line.detected = False
            right_line.detected = False
            left_line.recent_xfitted = []
            right_line.recent_xfitted = []
            left_line.allx = [] 
            right_line.allx = [] 
            left_line.ally = [] 
            right_line.ally = []
            
    return result,cte,curverad