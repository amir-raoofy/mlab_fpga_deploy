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
import gc # memory garbage collector
import pandas as pd

# Silence TensorFlow messages
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow.contrib.decent_q

from tensorflow.python.platform import gfile
from keras.preprocessing.image import img_to_array

from config import fcn_config as cfg
from config import fcn8_cnn   as cnn
import data_utils as dus


DATAS_DIR     = cfg.DATASET_DIR

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES
BATCH_SIZE = cfg.BATCH_SIZE

RAW_DATASET_DIR = cfg.RAW_DATASET_DIR
TEST_DIR = os.path.join(RAW_DATASET_DIR, 'train_v2')
dir_data = cfg.DATASET_DIR
b_test_csv = pd.read_csv(os.path.join(dir_data,"calib.csv"))

#purely vectorial python coding style
def softmax_predict(logits):
	prediction = tf.nn.softmax(logits, name='prediction')
	return prediction

#C-like coding style
def cpp_softmax(inpTens, H, W, Nclass) :
        result = np.zeros((H,W, Nclass), dtype="float32")
        iTens= inpTens
        for r in range(H): # loop on rows
                for c in range(W): # loop on columns
                        sum = 0.0
                        for d in range(Nclass): # loop on Classes
                                result[r,c,d]=np.exp(iTens[r,c,d])
                                sum = sum + result[r,c,d]
                        for d in range(Nclass): # loop on Classes
                                result[r,c,d]=result[r,c,d]/sum
        return result


def graph_eval(input_graph_def, input_node, output_node):
        #Reading images and segmentation labels
        #x_test, y_test, img_file, seg_file= cnn.get_images_and_labels(IMG_TEST_DIR, SEG_TEST_DIR,
        #                                                              cfg.NUM_CLASSES, cfg.WIDTH, cfg.HEIGHT)

        x_test, y_test = dus.batch_data_get(b_test_csv, TEST_DIR, cfg.BATCH_SIZE, augmentation=None)

        
        y_pred2  = np.zeros((y_test.shape))
        y_pred2i = np.zeros((y_test.shape[0],y_test.shape[1],y_test.shape[2]))

        # load graph
        tf.import_graph_def(input_graph_def,name = '')

        # Get input & output tensors
        x = tf.compat.v1.get_default_graph().get_tensor_by_name(input_node+':0')
        y = tf.compat.v1.get_default_graph().get_tensor_by_name(output_node+':0')

        # Create the Computational graph
        with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.initializers.global_variables())
                feed_dict={x: x_test}
		#y_pred = sess.run(y, feed_dict) # original code
                logits = sess.run(y, feed_dict)  # new code
                pred_DB = softmax_predict(logits) #new code
                y_pred = pred_DB.eval() #new code
                ##alternative way to compute softmax in C++ style
                #for i in range(logits.shape[0]):
                #        y_pred2[i]= cpp_softmax(logits[i], 224, 224, 12)


        # Calculate intersection over union for each segmentation class
        y_predi = np.argmax(y_pred, axis=3)
        y_testi = np.argmax(y_test, axis=3)
        print(y_testi.shape,y_predi.shape)
        cnn.IoU(y_testi,y_predi)

        ##just to check the cpp_softmax works fine
        #y_pred2i = np.argmax(y_pred2, axis=3)
        #print(y_testi.shape,y_pred2i.shape)
        #cfg.IoU(y_testi,y_pred2i)

        print ('FINISHED!')
        return 


def main(unused_argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
        input_graph_def = tf.Graph().as_graph_def()
        input_graph_def.ParseFromString(tf.io.gfile.GFile(FLAGS.graph, "rb").read())

        graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)





if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str,
                        default="../quantized_results/quantize_eval_model.pb",
                        help="graph file (.pb) to be evaluated.")
    parser.add_argument("--input_node", type=str,
                        default="input_1",
                        help="input node.")
    parser.add_argument("--output_node", type=str,
                        default="conv2d_transpose_3/conv2d_transpose",
                        help="output node.")
    parser.add_argument("--class_num", type=int,
                        default=cfg.NUM_CLASSES,
                        help="number of classes.")
    parser.add_argument("--gpu", type=str,
                        default="0",
                        help="gpu device id.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
