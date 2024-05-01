usr/bin/env python3

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from torchvision.io.video import re
from tqdm import tqdm
import csv
import timm
import wandb

from PIL import Image
import torchvision.transforms.v2 as transforms


from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display
import os
import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_dir = "/raid/datasets/hackathon2024"


mtcnn = MTCNN(keep_all=True, device=device)

wandb.login(key="c5c292dfefdac173c19a6d2234a73bf850d87aa5")

run = wandb.init(
            project="automathon"
            )

root_dir= os.path.expanduser("/raid/datasets/hackathon2024")
root_dir = os.path.join(root_dir, "test_dataset")
video_files = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]

out_path = "/raid/home/automathon_2024/account12/automathon-2024/output"
for nb,video_name in enumerate(video_files) :
    try:
        frames1 = []
        print("gogogogo")
        video = cv2.VideoCapture(root_dir+'/'+video_name)
        while True:
            read, frame= video.read()
            if not read:
                break
            frames1.append(frame)


        frames2 = np.array(frames1)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames2]

        frames_tracked = []
        boxes_frame =[]
        for i, frame in enumerate(frames):
            boxes, _ = mtcnn.detect(frame)
            boxes_frame.append(boxes.tolist())
            frame_draw = frame.copy()
            box = boxes[0]
            top,left,bot,right = box

            frames_tracked.append((frame_draw.crop(box.tolist()).resize((112, 112), Image.BILINEAR)))

        print("\nDone"+str(nb))

        out = cv2.VideoWriter(out_path+'/'+video_name[:-4]+''+'.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 30, (112,112))

        numpy_image = np.array(frames_tracked[0])
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        out.write(opencv_image)
        released = False
        i = 1
        while True:
            numpy_image = np.array(frames_tracked[i % len(frames_tracked)])
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            if not released :
                out.write(opencv_image)
            i += 1
            if not released and (i%len(frames_tracked)==0 or i%len(frames_tracked) == 1) :
                out.release()
                print("released")
                released = True
                break
    except KeyboardInterrupt:
        break
    except:
        print(video_name + "échec passage à la suivante")
        pass
