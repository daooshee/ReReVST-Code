import cv2
import glob
import os
import scipy.io as scio
import numpy as np
import random
import time

import torch

from framework import Stylization




## -------------------
##  Parameters


# Target style
style_img = './inputs/plum_flower.jpg'

# Target content video
# Use glob.glob() to search for all the frames
# Add sort them by sort()
content_video = './inputs/ambush_4/*.png'

# Path of the checkpoint (please download and replace the empty file)
checkpoint_path = "./Model/style_net-TIP-final.pth"

# Device settings, use cuda if available
cuda = torch.cuda.is_available()

# The proposed Sequence-Level Global Feature Sharing
use_Global = True

# Saving settings
save_video = True
fps = 24

# Where to save the results
result_frames_path = './result_frames/'
result_videos_path = './result_videos/'


## -------------------
##  Tools


if not os.path.exists(result_frames_path):
    os.mkdir(result_frames_path)

if not os.path.exists(result_videos_path):
    os.mkdir(result_videos_path)


def read_img(img_path):
    return cv2.imread(img_path)


class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                          64, self.record_W-64-W, cv2.BORDER_REFLECT)
        return new_img




## -------------------
##  Preparation


# Read style image
if not os.path.exists(style_img):
    exit('Style image %s not exists'%(style_img))
style = cv2.imread(style_img)

# Build model
framework = Stylization(checkpoint_path, cuda, use_Global)
framework.prepare_style(style)

# Read content frames
frame_list = glob.glob(content_video)

# Name for this testing
style_name = (style_img.split('/')[-1]).split('.')[0]
video_name = (content_video.split('/')[-2])
name = 'ReReVST-' + style_name + '-' + video_name
if not use_Global:
    name = name + '-no-global'

# Mkdir corresponding folders
if not os.path.exists('{}/{}'.format(result_frames_path,name)):
    os.mkdir('{}/{}'.format(result_frames_path,name))

# Build tools
reshape = ReshapeTool()




## -------------------
##  Inference


frame_num = len(frame_list)

# Prepare for proposed Sequence-Level Global Feature Sharing

if use_Global:

    print('Preparations for Sequence-Level Global Feature Sharing')
    framework.clean()
    interval = 8
    sample_sum = (frame_num-1)//interval
    
    for s in range(sample_sum):
        i = s * interval
        print('Add frame %d , %d frames in total'%(s, sample_sum))
        input_frame = read_img(frame_list[i])
        framework.add(input_frame)

    input_frame = read_img(frame_list[-1])
    framework.add(input_frame)

    print('Computing global features')
    framework.compute()

    print('Preparations finish!')

# Main stylization

for i in range(frame_num):

    print("Stylizing frame %d"%(i))

    # Read the image
    input_frame = read_img(frame_list[i])

    # Crop the image
    H,W,C = input_frame.shape
    new_input_frame = reshape.process(input_frame)

    # Stylization
    styled_input_frame = framework.transfer(new_input_frame)

    # Crop the image back
    styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

    # Save result
    cv2.imwrite('{}/{}/{}'.format(result_frames_path, name, 
                                  frame_list[i].split('/')[-1]), styled_input_frame)

# Write images back to video

if save_video:
    frame_list = glob.glob("{}/{}/*.*".format(result_frames_path,name))
    frame_list.sort()
    demo = cv2.imread(frame_list[0])
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Also (*'DVIX') or (*'X264')
    videoWriter = cv2.VideoWriter('{}/{}.avi'.format(result_videos_path, name), 
                                   fourcc, fps, (demo.shape[1],demo.shape[0]))

    for frame in frame_list:
        videoWriter.write(cv2.imread(frame))
    videoWriter.release()
