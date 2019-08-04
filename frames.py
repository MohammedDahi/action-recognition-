#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:53:05 2018
@author: mohammed
""" 
import cv2 
from os import listdir
from os.path import isfile, join
import os
import numpy as np
subfolders = [f.path for f in os.scandir("/home/mohammed/UCF101") if f.is_dir() ] 
for action_folder in subfolders :
    action=os.path.basename(os.path.normpath(action_folder))
    os.mkdir("/home/mohammed/action.test/"+action)
    for video in os.listdir(action_folder):
        if video in open('/home/mohammed/Downloads/ucfTrainTestlist/testlist01.txt').read():

            vidcap = cv2.VideoCapture(action_folder+"/"+video)
            count = 0
            test=False
            test,image = vidcap.read()
            print (action_folder+video)
            framecount=vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            framestep=np.divide(framecount,25)
            while test:
              cv2.imwrite("/home/mohammed/action.test/%s/%s_frame%d.jpg"%(action, video , count) , image)     # save frame as JPEG file
              count += framestep
              vidcap.set(cv2.CAP_PROP_POS_FRAMES,count )
              test,image = vidcap.read()
              print ('Read a new frame: ', test)
