#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:53:23 2018

@author: mohammed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import
import numpy as np
import tensorflow as tf

import os
from tensorflow.python.client import device_lib
import glob

import cv2

tf.logging.set_verbosity(tf.logging.INFO)
_NUM_CHANNELS = 3
_NUM_CLASSES = 101

tf.app.flags.DEFINE_string('output_directory', '/home/mohammed/output4', 'Output data directory')

FLAG = tf.app.flags.FLAGS

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print (local_device_protos)
    return [x.name  for x in local_device_protos if x.device_type == 'GPU']

def cnn_model_fn(features, labels, mode):


    with tf.device('/device:GPU:1'):


        input_layer = tf.reshape(features["image"], [-1, 224, 224, 3])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=96,
            kernel_size=[7, 7],
            padding="same",
            strides=2,
            activation=tf.nn.relu)

        norm1=tf.nn.local_response_normalization(conv1)
    

=
        pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)


        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=512,
            kernel_size=[5, 5],
            padding="same",
            strides=2,
            activation=tf.nn.relu)
        norm2=tf.nn.local_response_normalization(conv2)

        pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

         

         
         
         
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
    

 
        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
       

# 
            
        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        


        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)    
        print(pool5)
        # Dense Layer
        pool5_flat = tf.reshape(pool5, [-1, 7*7*512])
        dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(
            inputs=dense1, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(
            inputs=dense2, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
#        dense3 = tf.layers.dense(inputs=dropout2, units=1000, activation=tf.nn.relu)
#        dropout3 = tf.layers.dropout(
#            inputs=dense3, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
##    

        logits = tf.layers.dense(inputs=dropout2, units=101,activation=tf.nn.relu)
        #logits = tf.reshape(logits, [2, 2])
        
    
        predictions = {

            "classes": tf.argmax(input=logits, axis=1,name="class"),
           
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int64), depth=101)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}



        
        return tf.estimator.EstimatorSpec(
           mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

import matplotlib.pyplot as plt


def get_file_lists(data_dir):
    import glob

    train_list = glob.glob(data_dir)
    valid_list = glob.glob(data_dir)
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list




def train_input_fn(file_path):
    return input_fn(True, file_path, 100, None, 10)


def validation_input_fn(file_path):
    return input_fn(False, file_path, 50, 1, 1)


def _extract_features(example):
    features = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
                "image/class/label": tf.FixedLenFeature((), tf.int64),
    }
    parsed_example = tf.parse_single_example(example, features)
    image = tf.cast(tf.image.decode_jpeg(parsed_example["image/encoded"]), dtype=tf.float32)

#    image=tf.image.resize_image_with_crop_or_pad(
#    image,
#    128,
#    128)    
    images = tf.cast(image, tf.float32) * (1. / 255) 

    label = parsed_example['image/class/label']

    return images,label


def input_fn():
    files=glob.glob("/home/mohammed/tfrecords/train.aug/*")
    #here i used tfrecords to improve loading speed 
    file=["/home/mohammed/tfrecords/train.aug/train-00000-of-00001"]
    dataset=tf.data.TFRecordDataset(file) 
    dataset = dataset.map(_extract_features)
    dataset = dataset.batch(batch_size=128)
    dataset=dataset.repeat(2)
    iterator = dataset.make_one_shot_iterator()

    it = 0
    config=tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        next_images,name = iterator.get_next()
        label=sess.run(name)
        images = sess.run(next_images)

        
    return images,label


def input_fn_test():
    files=glob.glob("/home/mohammed/tfrecords/test.aug/*")

    file=["/home/mohammed/tfrecords/test.aug/train-00000-of-00001"]
    dataset=tf.data.TFRecordDataset(file) 
    dataset = dataset.map(_extract_features)
    dataset = dataset.batch(batch_size=1)

    iterator = dataset.make_one_shot_iterator()

    it = 0
    config=tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        next_images,name = iterator.get_next()
        label=sess.run(name)
        images = sess.run(next_images)

        
    return images,label
def main(unused_arg):
    get_available_gpus()
    
#    train_list, valid_list = get_file_lists("/home/mohammed/non/train-00000-of-00001")

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=os.path.join(FLAG.output_directory, "tb"))

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    

    input_fnn = tf.estimator.inputs.numpy_input_fn(  x={'image':input_fn()[0]}, y=input_fn()[1]
    ,
    num_epochs=2,
    shuffle=True)
    
    input_fnn_test = tf.estimator.inputs.numpy_input_fn(  x={'image':input_fn_test()[0]}, y=input_fn_test()[1]
    ,
    num_epochs=1,
    shuffle=False)
    
    train_spec = tf.estimator.TrainSpec(input_fn=input_fnn)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fnn_test)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

 #   classifier.train(input_fnn,steps = 1, hooks=[logging_hook])
    
                    
    #evalution = classifier.evaluate(input_fnn)

    #print(evalution)
   # results = classifier.predict(input_fnn)

#    for p in results:
#        #print(p['probabilities'])
# 
#        print(p['classes'])


    


if __name__ == '__main__':
    tf.app.run()
