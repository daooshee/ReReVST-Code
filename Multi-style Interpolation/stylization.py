from __future__ import print_function
import os
import random

import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.utils as vutils

import numpy as np
import cv2

def transform_image(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1))).float()
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

def transform_back_image(img):
    img = img.data.cpu()
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img*std+mean
    img = img.clamp(0, 1)[0,:,:,:]
    img =  img.numpy().transpose((1, 2, 0))
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR) * 255

def transform_data(data):
    data = torch.from_numpy(data.transpose((2, 0, 1))).float()
    return data.unsqueeze(0)

def transform_back_data(img):
    img = img.data.cpu()
    img = img.clamp(0, 1)[0,:,:,:]
    img =  img.numpy().transpose((1, 2, 0))
    return img * 255

class Stylization():
    def __init__(self, checkpoint='', cuda=True, style_num=1):
        print("Style Num:", style_num)

        # ===== Prepare GPU and Seed =====
        if cuda:
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        # ===== Transfer Net =====
        from style_network import TransformerNet

        if os.path.exists(checkpoint):
            self.transformer = TransformerNet(style_num=style_num).to(self.device)
            self.transformer.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
        else:
            exit("Cannot find TransformerNet checkpoints.")

        for param in self.transformer.parameters():
            param.requires_grad = False

    def add_patch(self, patch_feature):
        self.transformer.add_patch(patch_feature)

    def prepare_style(self, style_images):
        if not self.transformer.have_delete_vgg:
            del self.transformer.Vgg19
            self.transformer.have_delete_vgg = True

        with torch.no_grad():
            for style_id, style_image in enumerate(style_images):
                style_image = transform_image(style_image).to(self.device).detach()
                self.transformer.generate_style_features(style_image, style_id)

    def compute_norm(self):
        with torch.no_grad():
            self.transformer.compute_norm()

    def clean(self):
        self.transformer.clean()

    def generate_content_features(self, content):
        # Transform images into tensors
        with torch.no_grad():
            content = transform_image(content).to(self.device).detach()
            F_content = self.transformer.generate_content_features(content)

        return F_content

    def transfer(self, cur_feature, style_weight=[1.]):
        # Stylization
        with torch.no_grad():
            styled_cur_frame = self.transformer(cur_feature, style_weight).detach()
            cur_frame_result = transform_back_image(styled_cur_frame)

        return cur_frame_result




