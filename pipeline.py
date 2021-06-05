import numpy as np
import cv2


# Useful functions for producing the binary pixel of interest images to feed into the LaneTracker algorithm

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

def abs_sobel_thresh(sobel, thresh=(0, 255)):
    # Calculate directional gradient
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(sobelx, sobely, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gradmag = np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # Apply threshold
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(sobelx, sobely, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output = np.zeros_like(sobelx)
        # Apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_threshold(image_channel, thresh=(0,255)):
    binary_output = np.zeros_like(image_channel)
    binary_output[(image_channel >= thresh[0]) & (image_channel <= thresh[1])] = 1
    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Edit this function to create your own pipeline.
# Note: This expects RGB images
def pipeline(image, l_thresh=(210, 255), b_thresh=(170, 255), grad_thresh=(30, 255), dir_thresh=(0, np.pi*0.40)):
    # Convert to LAB color space and separate the L and B channels
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]
    
    # Adaptive histogram equalization to compensate lighting conditions
    l_channel = clahe.apply(l_channel)
    
    # Choose a Sobel kernel size
    sobel_kernel = 11 # Choose a larger odd number to smooth gradient measurements

    # Calculating sobelx and sobely once to speed up processing
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
     
    # process image and generate binary pixel of interests

    # Threshold gradient magnitude
    grad_mag = mag_thresh(sobelx, sobely, mag_thresh=grad_thresh)
    # Threshold gradient direction
    grad_dir = dir_threshold(sobelx, sobely, thresh=dir_thresh)
    
    # Threshold color channel for white lanes
    l_binary = color_threshold(l_channel, thresh=l_thresh)
    
    # Threshold color channel for yellow lanes
    b_binary = color_threshold(b_channel, thresh=b_thresh)

    # Combine the binary thresholds
    combined_binary = np.zeros_like(image[:,:,0])
    combined_binary[((grad_mag==1)&(grad_dir==1)&(l_binary==1))|(b_binary == 1)] = 255

    return combined_binary

