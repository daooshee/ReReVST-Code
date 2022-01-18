### Consistent Video Style Transfer via Relaxation and Regularization 

Wenjing Wang, Shuai Yang, Jizheng Xu, and Jiaying Liu. "Consistent Video Style Transfer via Relaxation and Regularization", IEEE Trans. on Image Processing (TIP), 2020.

Project Website: https://daooshee.github.io/ReReVST/


[Update Jan. 2022] Test content videos and style images can be found in [./data](data)

[Update June 2021] Code for multi-style interpolation can be found in [./Multi-style Interpolation](Multi-style%20Interpolation) (but the code is messy).




<hr>


##### Prerequisites

* torch
* torchvision
* scipy
* numpy
* opencv-python-headless



#### 1. Training Code

```
cd ./train
```

##### 1.1 Data preparation

Download style images ([WikiArt.org](https://www.wikiart.org/) or [Painter by Numbers from Kaggle](https://www.kaggle.com/c/painter-by-numbers)) in `./data/style/`:
```
./data/style/1.jpg
./data/style/2.jpg
....
./data/style/99.jpg 
```

Download content images ([MSCOCO](https://cocodataset.org)) in `./data/content/`:
```
./data/content/COCO_train2014_000000000000.jpg
./data/content/COCO_train2014_000000000001.jpg
....
./data/content/COCO_train2014_000000000099.jpg 
```

##### 1.2 Training

Directly training the model with all the loss terms is hard, so we gradually add the loss terms and adjust the hyper-parameters.

The script below is based on a half pre-trained `style_net-epoch-0.pth`. If you are interested in training from scratch, please refer to the Supplementary Material, Sec. I. B. for detailed parameter settings.

**1.2.1 Download half pre-trained model**

Download the half pre-trained model and replace the empty `style_net-epoch-0.pth`.

Links: [Google Drive](https://drive.google.com/drive/folders/1RSmjqZTon3QdxBUSjZ3siGIOwUc-Ycu8?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1Td30bukn2nc4zepmSDs1mA) [397w]

**1.2.2 Training**

Please refer to `train.py` for the meaning of each parameter.

```
python3 train.py --cuda --gpu 0 --epoches 2 --batchSize 4 --lr 1e-4 --dynamic_filter --both_sty_con --relax_style --style_content_loss --contentWeight 1 --recon_loss --tv_loss --temporal_loss --data_sigma --data_w
```



#### 2. Testing Code

```
cd ./test
```

##### 2.1 Download model

Download our final model and replace the empty `style_net-TIP-final.pth`.

Links: [Google Drive](https://drive.google.com/drive/folders/1RSmjqZTon3QdxBUSjZ3siGIOwUc-Ycu8?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1Td30bukn2nc4zepmSDs1mA) [397w]

##### 2.2 Inference

In `generate_real_video.py`, set the path of the style image by `style_img`, and set the path to the video frames by `content_video`, and run:

```
python generate_real_video.py
```

Then, you can find stylized videos in `./result_frames/` and `./result_videos/`. When writing the final video, the frames will be sorted by their names (using  `sort()` in python).

