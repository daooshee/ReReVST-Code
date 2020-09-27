import os 
import cv2
import random
import numpy as np
import glob
import scipy.io as scio
import pickle
import io
import zipfile

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.utils as vutils




Min_Max_VGG = [[-2.11790393, 2.2489083 ], [-2.03571429, 2.42857143], [-1.80444444, 2.64]]




def warp_opencv(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img.astype(np.float32), flow.astype(np.float32), None, cv2.INTER_LINEAR)
    return res




class VideoDataset(data.Dataset):
    
    ''' This is for the ablation study in Figure 16. 
        It might have some bugs, but I'm tired of cleaning up this code (●ﾟωﾟ●)
        If you really need this class, you can remind me to clean it up by <Issues>.
    '''
    
    def __init__(self, loadSize=288, fineSize=256, flip=True, 
                    video_path="data/Video", style_path="data/style", data=None):
        super(VideoDataset, self).__init__()
        # the Flow field should be in the range [-1,1] assuming normalized co-ordinates of the image/video.
        # https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/6

        # Data Lists

        # The video_path should have:
        # # (/*/*/*.jpg) frames
        # # (flow_mat/) optical flows
        # # (occlusions/) occlusion maps

        self.video_zip_path     = video_path
        self.pre_frame_list     = data['pre_frame_list'] # frame1, frame2, frame3 ... 
        self.cur_frame_list     = data['cur_frame_list'] # frame2, frame3, frame4 ... 
        self.flow_list          = data['flow_list']  # flow1_2, flow2_3, ...
        self.mask_list          = data['mask_list']  # mask1_2, mask2_3, ...

        # Style data
        if style_path.split('.')[-1] == 'zip':
            self.use_zip = True

            self.style_zip = zipfile.ZipFile(style_path)
            style_img_list = self.style_zip.namelist()

            self.style_img_list = []
            for img_path in style_img_list:
                if img_path[-4:] == '.jpg':
                    self.style_img_list.append(img_path)

            self.style_zip = style_path
        else:
            self.use_zip = False
            self.style_img_list = glob.glob(style_path+"/*.jpg")

        self.style_img_list_len = len(self.style_img_list)

        # Parameters
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip


    def ProcessImg(self, img, size=None, x1=None, y1=None, flip_rand=None):
        ''' Given an image with channel [BGR] which values [0,255]
            The output values [-1,1]
        '''
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if size is not None:
            img = cv2.resize(img,(size, size))
        img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if self.flip == 1:
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img.div_(255.0)
        img = (img - mean) / std

        return img

    def ProcessMask(self, img, x1, y1, flip_rand):
        ''' Given a numpy '''
        img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if self.flip == 1:
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        return img

    def ProcessFlow(self, img, x1, y1, flip_rand):
        ''' Given a numpy '''
        img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if self.flip == 1:
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
                img[:,:,0] = -img[:,:,0]
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
                img[:,:,1] = -img[:,:,1]
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)
                img[:,:,0] = -img[:,:,0]
                img[:,:,1] = -img[:,:,1]

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        return img

    def __getitem__(self, index):
        ###############################
        # Read Data
        # -----------------------------

        # Read MPI data
        with zipfile.ZipFile(self.video_zip_path) as video_zip:
            pre_frame = video_zip.read(self.pre_frame_list[index])
            pre_frame = cv2.imdecode(np.frombuffer(pre_frame, np.uint8), 1)

            cur_frame = video_zip.read(self.cur_frame_list[index])
            cur_frame = cv2.imdecode(np.frombuffer(cur_frame, np.uint8), 1)

            forwardflow = video_zip.read(self.flow_list[index])
            forwardflow = np.frombuffer(forwardflow, np.float32)
            forwardflow = np.reshape(forwardflow[32:], (cur_frame.shape[0],cur_frame.shape[1],2))

            mask = video_zip.read(self.mask_list[index])
            mask = cv2.imdecode(np.frombuffer(mask, np.uint8), 1)
            mask = 1 - mask / 255.

        # Read style data
        if self.use_zip:
            # Read style image
            # Have to build a new zipfile.ZipFile each time for parallel processing
            style_zip = zipfile.ZipFile(self.style_zip)
            style_img_path = self.style_img_list[random.randint(0,self.style_img_list_len-1)]
            style_img_path = style_zip.read(style_img_path)
            
            style_img = cv2.imdecode(np.frombuffer(style_img_path, np.uint8), 1)

        else:
            # Read style image
            style_img_path = self.style_img_list[random.randint(0,self.style_img_list_len-1)]
            style_img = cv2.imread(style_img_path)

        ###############################
        # Process Data
        # -----------------------------

        Data = {}

        ###############################
        # Process MPI Data
        # -----------------------------

        x1 = random.randint(0, pre_frame.shape[0]-self.fineSize)
        y1 = random.randint(0, pre_frame.shape[1]-self.fineSize)
        flip_rand = random.random()

        Data['Content']         = self.ProcessImg(pre_frame,None,x1,y1,flip_rand)
        Data['NextContent']     = self.ProcessImg(cur_frame,None,x1,y1,flip_rand)
        Data['ForwardFlow']    = self.ProcessFlow(forwardflow,x1,y1,flip_rand)
        Data['ForwardMask']    = self.ProcessMask(mask,x1,y1,flip_rand)

        ###############################
        # Process Style Image
        # -----------------------------

        H,W,C = style_img.shape
        loadSize = max(H, W, self.loadSize)

        x1 = random.randint(0, loadSize - self.fineSize)
        y1 = random.randint(0, loadSize - self.fineSize)
        flip_rand = random.random()

        Data['Style'] = self.ProcessImg(style_img, loadSize, x1, y1, flip_rand)
        return Data

    def __len__(self):
        return len(self.cur_frame_list)




class MPIDataset(data.Dataset):

    ''' This is for the ablation study in Figure 16. 
        It might have some bugs, but I'm tired of cleaning up this code (●ﾟωﾟ●)
        If you really need this class, you can remind me to clean it up by <Issues>.
    '''

    def __init__(self, loadSize=288, fineSize=256, flip=True, 
                    mpi_path="data/MPI", style_path="data/style"):
        super(MPIDataset, self).__init__()
        # the Flow field should be in the range [-1,1] assuming normalized co-ordinates of the image/video.
        # https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/6

        # Data Lists

        # The mpi_path should have:
        # # (clean/) frames
        # # (flow_mat/) optical flows
        # # (occlusions/) occlusion maps

        self.pre_frame_list = [] # frame1, frame2, frame3 ... 
        self.cur_frame_list = [] # frame2, frame3, frame4 ... 
        self.flow_list = []  # flow1_2, flow2_3, ...
        self.mask_list = []  # mask1_2, mask2_3, ...

        Folder = glob.glob(mpi_path+"/clean/*/")

        for folder in Folder:
            ImgNumber = len(glob.glob(folder+"*.png"))
            name_folder = folder.split("/")[-2]

            for i in range(2,ImgNumber+1):
                assert os.path.exists(folder+"frame_%04d.png"%(i-1))
                self.pre_frame_list.append(folder+"frame_%04d.png"%(i-1))

                assert os.path.exists(folder+"frame_%04d.png"%(i))
                self.cur_frame_list.append(folder+"frame_%04d.png"%(i))

                assert os.path.exists("%s/flow_mat/%s_frame_%04d.mat"%(mpi_path,name_folder,i-1))
                self.flow_list.append("%s/flow_mat/%s_frame_%04d.mat"%(mpi_path,name_folder,i-1))

                assert os.path.exists(mpi_path+"/occlusions/"+name_folder+"/frame_%04d.png"%(i-1))
                self.mask_list.append(mpi_path+"/occlusions/"+name_folder+"/frame_%04d.png"%(i-1))


        # Style data
        if style_path.split('.')[-1] == 'zip':
            self.use_zip = True

            self.style_zip = zipfile.ZipFile(style_path)
            style_img_list = self.style_zip.namelist()
            self.style_img_list = []
            for img_path in style_img_list:
                if img_path[-4:] == '.jpg':
                    self.style_img_list.append(img_path)

            self.style_zip = style_path
        else:
            self.use_zip = False
            self.style_img_list = glob.glob(style_path+"/*.jpg")

        self.style_img_list_len = len(self.style_img_list)


        # Parameters
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip


    def ProcessImg(self, img, size=None, x1=None, y1=None, flip_rand=None):
        ''' Given an image with channel [BGR] which values [0,255]
            The output values [-1,1]
        '''
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if size is not None:
            img = cv2.resize(img,(size, size))
        img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if self.flip == 1:
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img.div_(255.0)
        img = (img - mean) / std

        return img

    def ProcessMask(self, img, x1, y1, flip_rand):
        ''' Given a numpy '''
        img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if self.flip == 1:
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        return img

    def ProcessFlow(self, img, x1, y1, flip_rand):
        ''' Given a numpy '''
        img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if self.flip == 1:
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
                img[:,:,0] = -img[:,:,0]
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
                img[:,:,1] = -img[:,:,1]
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)
                img[:,:,0] = -img[:,:,0]
                img[:,:,1] = -img[:,:,1]

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        return img

    def __getitem__(self, index):
        ###############################
        # Read Data
        # -----------------------------

        # Read MPI data
        pre_frame = cv2.imread(self.pre_frame_list[index])
        cur_frame = cv2.imread(self.cur_frame_list[index])
        backwardflow = - scio.loadmat(self.flow_list[index],verify_compressed_data_integrity=False)['Img']
        mask = 1 - cv2.imread(self.mask_list[index]) / 255.
        mask *= warp_opencv(np.ones([436, 1024,3]), backwardflow)

        # Read style data
        if self.use_zip:
            # Read style image
            # Have to build a new zipfile.ZipFile each time for parallel processing
            style_zip = zipfile.ZipFile(self.style_zip)
            style_img_path = self.style_img_list[random.randint(0,self.style_img_list_len-1)]
            style_img_path = style_zip.read(style_img_path)
            
            style_img = cv2.imdecode(np.frombuffer(style_img_path, np.uint8), 1)

        else:
            # Read style image
            style_img_path = self.style_img_list[random.randint(0,self.style_img_list_len-1)]
            style_img = cv2.imread(style_img_path)

        ###############################
        # Process Data
        # -----------------------------

        Data = {}

        ###############################
        # Process MPI Data
        # -----------------------------

        x1 = random.randint(0, pre_frame.shape[0]-self.fineSize)
        y1 = random.randint(0, pre_frame.shape[1]-self.fineSize)
        flip_rand = random.random()

        Data['Content']         = self.ProcessImg(pre_frame,None,x1,y1,flip_rand)
        Data['NextContent']     = self.ProcessImg(cur_frame,None,x1,y1,flip_rand)
        Data['BackwardFlow']    = self.ProcessFlow(backwardflow,x1,y1,flip_rand)
        Data['BackwardMask']    = self.ProcessMask(mask,x1,y1,flip_rand)

        ###############################
        # Process Style Image
        # -----------------------------

        H,W,C = style_img.shape
        loadSize = max(H, W, self.loadSize)

        x1 = random.randint(0, loadSize - self.fineSize)
        y1 = random.randint(0, loadSize - self.fineSize)
        flip_rand = random.random()

        Data['Style'] = self.ProcessImg(style_img, loadSize, x1, y1, flip_rand)
        return Data

    def __len__(self):
        return len(self.cur_frame_list)




class FrameDataset(data.Dataset):

    ''' The dataloader for the final training. '''

    def __init__(self, loadSize=288, fineSize=256, flip=True, 
                        content_path="data/content", style_path="data/style"):
        super(FrameDataset, self).__init__()

        # the Flow field should be in the range [-1,1] assuming normalized co-ordinates of the image/video.
        # https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/6

        # Data Lists
        self.content_img_list = glob.glob(content_path+"/*.jpg")
        self.style_img_list = glob.glob(style_path+"/*.jpg")
        self.style_img_list_len = len(self.style_img_list)

        # Parameters
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip


    def ProcessImg(self, img, size=None, x1=None, y1=None, flip_rand=None):
        ''' Given an image with channel [BGR] which values [0,255]
            The output values [-1,1]
        '''
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        h,w,c = img.shape

        if(size != None):
            img = cv2.resize(img,(size, size))
            img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if(self.flip == 1):
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img.div_(255.0)
        img = (img - mean) / std

        return img


    def Process(self, input):
        ''' Given a numpy '''
        return torch.from_numpy(input.transpose((2, 0, 1))).float()


    def RandomBlur(self, img):
        img = img.data.numpy().transpose((1, 2, 0))
        H,W,C = img.shape
        img = cv2.resize(img,(H+random.randint(-5,5), W+random.randint(-5,5)))
        img = cv2.resize(img,(H,W))
        return self.Process(img)


    def __getitem__(self, index):
        ###############################
        # Read Data
        # -----------------------------

        # Read content image
        first_img = self.content_img_list[index]
        first_frame = cv2.imread(first_img)

        # Read style image
        style_img_path = self.style_img_list[random.randint(0,self.style_img_list_len-1)]
        style_img = cv2.imread(style_img_path)

        Sequence = {}

        ###############################
        # Process Content Image
        # -----------------------------

        x1 = random.randint(0, self.loadSize - self.fineSize)
        y1 = random.randint(0, self.loadSize - self.fineSize)
        flip_rand = random.random()

        first_frame = self.ProcessImg(first_frame, self.loadSize, x1, y1, flip_rand)
        Sequence['Content'] = first_frame

        ###############################
        # Process Style Image
        # -----------------------------

        H,W,C = style_img.shape
        loadSize = max(H, W, self.loadSize)

        x1 = random.randint(0, loadSize - self.fineSize)
        y1 = random.randint(0, loadSize - self.fineSize)
        flip_rand = random.random()

        Sequence['Style'] = self.ProcessImg(style_img, loadSize, x1, y1, flip_rand)

        return Sequence

    def __len__(self):
        return len(self.content_img_list)




def get_loader(batch_size, loadSize=288, fineSize=256, flip=True, 
                    content_path="./data/content/", style_path="./data/style/", num_workers=16, use_mpi=False, use_video=False):
    
    if use_mpi and use_video:
        exit('Find both use_mpi and use_video, can only use one of them at the same time.')

    if use_mpi:
        dataset = MPIDataset(loadSize=loadSize, fineSize=fineSize, flip=flip, 
                                mpi_path=content_path, style_path=style_path)
    elif use_video:
        with open('video_data.pickle', 'rb') as f:
            video_data = pickle.load(f)

        dataset = VideoDataset(loadSize=loadSize, fineSize=fineSize, flip=flip, 
                                video_path=content_path, style_path=style_path, 
                                data=video_data)
    else:
        dataset = FrameDataset(loadSize=loadSize, fineSize=fineSize, flip=flip, 
                                content_path=content_path, style_path=style_path)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader



