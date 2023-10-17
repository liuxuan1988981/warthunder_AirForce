# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np
import config
from src.YOLO import YOLO

from utils.misc_utils import load_weights

tf.compat.v1.disable_eager_execution()

weight_path = './yolo_weights/yolov4.weights'
save_path = './yolo_weights/yolov4.ckpt'
anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]       # for yolov4.weights
class_num = 80      # for yolov4.weights

yolo = YOLO()
with tf.compat.v1.Session() as sess:
    inputs = tf.compat.v1.placeholder(tf.float32, [1, 608, 608, 3])

    feature = yolo.forward(inputs, class_num, isTrain=False)
    
    # for var_name in tf.compat.v1.global_variables():
    #     print(var_name.name)
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())

    load_ops = load_weights(tf.compat.v1.global_variables(), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
    

