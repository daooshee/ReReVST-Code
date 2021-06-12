import PIL.Image, PIL.ImageTk
import time
import os
import glob

import cv2
import numpy as np
from matplotlib import cm
import torch

from stylization import Stylization

# https://www.geeksforgeeks.org/python-gui-tkinter/
# based on: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/

class ChangeShapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def Process(self, img):
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

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H, 64, self.record_W-64-W, cv2.BORDER_REFLECT)
        return new_img


class VideoStylization:
	def __init__(self, checkpoint="./", style_img_pair=[], test_sequence="", img_type="png"):
		cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if cuda else "cpu")

		print('Init Transfer Model.')
		self.transformer = Stylization(checkpoint=glob.glob(checkpoint+"*.pth")[0], cuda=cuda, style_num=len(style_img_pair))

		print('Load Style Images.')
		style_images = []
		for style_img in style_img_pair:
			if not os.path.exists(style_img):
				exit('%s not exists'%(style_img))
			style_images.append(cv2.resize(cv2.imread(style_img), (384,384)))

		print('Prepare Style Features.')
		self.transformer.prepare_style(style_images)

		frame_list = glob.glob(test_sequence+"/*.%s"%(img_type))
		frame_list.sort()

		self.frame_num = len(frame_list)
		assert(self.frame_num > 1)
		self.frame_list = frame_list

		self.ChangeShape = ChangeShapeTool()

		self.ContentFeaturePrePare()
		self.SeqNormPrePare()

		self.frame_id = 0

	def SeqNormPrePare(self, interval=16):
		self.transformer.clean()

		sample_num = (self.frame_num-1)//interval+1
		for s in range(sample_num):
			i = s*interval
			print('frame %d, %d frames in total'%(s, sample_num+1))
			tmp_frame = torch.load('cache/%d.pt'%(i)).to(self.device)
			self.transformer.add_patch(tmp_frame)

		print('frame %d, %d frames in total'%(sample_num, sample_num+1))
		tmp_frame = torch.load('cache/%d.pt'%(self.frame_num-1)).to(self.device)
		self.transformer.add_patch(tmp_frame)
		self.transformer.compute_norm()

	def ContentFeaturePrePare(self):
		if os.path.exists("cache"):
			os.system("rm -r cache")
		os.mkdir("cache")

		for frame_id, frame_path in enumerate(self.frame_list):
			print("F_content: %d/%d"%(frame_id, len(self.frame_list)))
			frame = cv2.imread(frame_path)
			H,W,C = frame.shape
			frame = self.ChangeShape.Process(frame)
			frame_feature = self.transformer.generate_content_features(frame)
			torch.save(frame_feature, 'cache/%d.pt'%(frame_id))

		self.H = H
		self.W = W

	def get_frame(self, weight):
		new_frame = torch.load('cache/%d.pt'%(self.frame_id)).to(self.device)
		self.frame_id += 1

		if self.frame_id == self.frame_num:
			self.frame_id = 0

		styled_frame = self.transformer.transfer(new_frame, weight)
		return (True, styled_frame[64:64+self.H,64:64+self.W,:])


style_img_pair_list = [["A.jpg", "B.jpg"]]
test_sequence_list = ["AAA"]

for style_id, style_img_pair in enumerate(style_img_pair_list):
	for test_sequence in test_sequence_list:
		save_path = "result/%s_%s"%(test_sequence, style_id)
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		vid = VideoStylization(style_img_pair=style_img_pair, test_sequence=test_sequence)

		image_num = len(glob.glob(test_sequence+"/*.png"))

		for i in range(image_num):
			weight_1 = i/(image_num-1.)
			print('Weight:', weight_1)
			ret, frame = vid.get_frame([weight_1, (1-weight_1)])
			cv2.imwrite(save_path+"/%d.png"%(i), frame)







