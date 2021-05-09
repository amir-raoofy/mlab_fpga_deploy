#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
/**

* © Copyright (C) 2016-2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
'''

# modified by daniele.bagni@xilinx.com
# date 20 / 11 / 2020


##################################################################
# Evaluation of frozen/quantized graph
#################################################################

import os
import sys
import glob
import argparse
import shutil
import tensorflow as tf
import numpy as np
import cv2
import gc # memory garbage collector #DB
import pandas as pd
import data_utils as dus

import matplotlib.pyplot as plt
import random


# reduce TensorFlow messages in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import tensorflow.contrib.decent_q

from tensorflow.python.platform import gfile
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend               import set_session


from config import fcn_config as cfg
from config import fcn8_cnn   as cnn

def get_script_directory():
    path = os.getcwd()
    return path

DATAS_DIR     = cfg.DATASET_DIR
IMG_TEST_DIR  = cfg.dir_test_img
SEG_TEST_DIR  = cfg.dir_test_seg

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES
BATCH_SIZE = cfg.BATCH_SIZE

RAW_DATASET_DIR = cfg.RAW_DATASET_DIR
TEST_DIR = os.path.join(RAW_DATASET_DIR, 'train_v2')
dir_data = cfg.DATASET_DIR
b_test_csv = pd.read_csv(os.path.join(dir_data,"calib.csv"))

def graph_eval(input_graph_def, input_node, output_node):
    #Reading images and segmentation labels
    #x_test, y_test, img_file, seg_file = cnn.get_images_and_labels(IMG_TEST_DIR, SEG_TEST_DIR, cfg.NUM_CLASSES, cfg.WIDTH, cfg.HEIGHT)

    x_test, y_test = dus.batch_data_get(b_test_csv, TEST_DIR, cfg.BATCH_SIZE, augmentation=None)

    # load graph
    tf.import_graph_def(input_graph_def,name = '')

    # Get input & output tensors
    x = tf.compat.v1.get_default_graph().get_tensor_by_name(input_node+':0')
    y = tf.compat.v1.get_default_graph().get_tensor_by_name(output_node+':0')

    # Create the Computational graph
    y_pred=np.zeros(x_test.shape)
    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.initializers.global_variables())
        feed_dict={x: x_test} #, labels: y_test}
        y_pred = sess.run(y, feed_dict)
        #for idx in range(0, x_test.shape[0], 32):
        #    sess.run(tf.compat.v1.initializers.global_variables())
        #    feed_dict={x: x_test[idx*32:(idx+1)*32]} #, labels: y_test}
        #    y_pred[idx*32:(idx+1)*32] = sess.run(y, feed_dict)


    # Calculate intersection over union for each segmentation class
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(y_test, axis=3)
    #print(y_testi.shape,y_predi.shape)


    n_samples = 20
    print (y_test.shape, y_pred.shape)
    for idx,jdx in enumerate (random.sample(range(0,y_pred.shape[0]), n_samples)):
        cnn.IoU(y_test[jdx],y_pred[jdx])
    
    #print(y_test.shape,y_pred.shape)
    fig, axs = plt.subplots(ncols=3, nrows=n_samples, figsize=(5, 25), sharex=True, sharey=True)
    for idx,jdx in enumerate (random.sample(range(0,y_pred.shape[0]), n_samples)):
        axs[idx,0].imshow(x_test[jdx,:,:])
        axs[idx,1].imshow(y_test[jdx,:,:,0])
        axs[idx,2].imshow(y_pred[jdx,:,:,0])
        axs[0,0].set_title('Input')
        axs[0,1].set_title('Mask')
        axs[0,2].set_title('Prediction')

        plt.xticks([])
        plt.yticks([])
        plt.savefig("../rpt/tmp9.png")
        plt.show()

    print ('FINISHED!')
    #return x_test, y_testi, y_predi, img_file, seg_file


def main(unused_argv):

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    set_session(tf.compat.v1.Session(config=config))

    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.io.gfile.GFile(FLAGS.graph, "rb").read())
    #x_test,y_testi,y_predi,img_file,seg_file = graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)
    graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str,
                        default="../freeze/frozen_graph.pb",
                        help="graph file (.pb) to be evaluated.")
    parser.add_argument("--input_node", type=str,
                        default="input_1",
                        help="input node.")
    parser.add_argument("--output_node", type=str,
                        default="activation_1/truediv",
                        help="output node.")
    parser.add_argument("--class_num", type=int,
                        default=cfg.NUM_CLASSES,
                        help="number of classes.")
    parser.add_argument("--gpu", type=str,
                        default="0",
                        help="gpu device id.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)