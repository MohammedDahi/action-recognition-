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
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
subfolders = [f.path for f in os.scandir("/home/mohammed/DeepLearning/BalancedDataset/train") if f.is_dir() ] 
for action_folder in subfolders :
    action=os.path.basename(os.path.normpath(action_folder))
    os.mkdir("/home/mohammed/DeepLearning/dataset_aug/train/"+action)
    for video in os.listdir(action_folder):
        if (action=="0") :
            im = cv2.imread(action_folder+"/"+video)
        
            
            im=cv2.resize(im, (460, 460))
        
                 
            test1 = im[0:224, 0:224]
            test2 = im[0:224, 32:256]
            test3 = im[32:256, 32:256]
            test4 = im[32:256, 0:224]
            testc = im[16:240, 16:240]
                 
            ftest1 = cv2.flip( test1, 0 )
            ftest2 = cv2.flip( test2, 0 )
            ftest3 = cv2.flip( test3, 0 )
            ftest4 = cv2.flip( test4, 0 )
            ftestc = cv2.flip( testc, 0 )
        
        
        
            count = 0
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test1)     # save frame as JPEG file
            count = 1
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test2)     # save frame as JPEG file
            count = 2
                 
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test3)     # save frame as JPEG file
            count = 3
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test4) 
            count=4 # save frame as JPEG file
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , testc)     # save frame as JPEG file
                 
                 
                 
            count = 5
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest1)     # save frame as JPEG file
            count = 6
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest2)     # save frame as JPEG file
            count = 7
                 
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest3)     # save frame as JPEG file
            count = 8
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest4) 
            count=9 # save frame as JPEG file
            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftestc)     # save frame as JPEG file
            
            print ('Read a new frame: ', count)
