# coding:utf-8
# process yolo 
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   #设置为1屏蔽一般信息，2屏蔽一般和警告，3屏蔽所有输出

import numpy as np
from utils import tools
import cv2
import config
from src import Log
import time

class_num = config.voc_class_num
width = config.width
height = config.height
anchors =  np.asarray(config.voc_anchors).astype(np.float32).reshape([-1, 3, 2])
score_thresh = config.val_score_thresh
iou_thresh = config.val_iou_thresh
max_box = config.max_box
model_path = config.voc_model_path
name_file = config.voc_names
val_dir = config.voc_test_dir
save_img = config.save_img
save_dir = config.voc_save_dir

# dictionary of name of corresponding id
word_dict = tools.get_word_dict(name_file)
# dictionary of per names
color_table = tools.get_color_table(class_num)

# tf session
sess = None
pre_boxes, pre_score, pre_label, inputs = [None]*4

def deal_img(img_ori):
    if img_ori is None:
        return None, None, None, None, None
    ori_h, ori_w, _ = img_ori.shape
    img = cv2.resize(img_ori, (width, height))

    show_img = img
    
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, ori_w, ori_h, img_ori, show_img

# 初始化函数
def init_func():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    configTf = ConfigProto()
    configTf.gpu_options.allow_growth = True
    session = InteractiveSession(config=configTf)
    
    import tensorflow as tf
    from src.YOLO import YOLO
    from src.Feature_parse_tf import get_predict_result

    tf.compat.v1.disable_eager_execution()
    tf = tf.compat.v1
    print("====== init yolo object ======")
    yolo = YOLO()
    global inputs
    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    print("====== init forward ======")
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, class_num, isTrain=False)
    global pre_boxes, pre_score, pre_label 
    print("====== init result predict ======")
    pre_boxes, pre_score, pre_label = get_predict_result(feature_y1, feature_y2, feature_y3,
                                                                                                anchors[2], anchors[1], anchors[0], 
                                                                                                width, height, class_num, 
                                                                                                score_thresh=score_thresh, 
                                                                                                iou_thresh=iou_thresh,
                                                                                                max_box=max_box)

    init = tf.compat.v1.global_variables_initializer()
    print("====== init Saver ======")
    saver = tf.train.Saver()

    print("====== init session ======")
    global sess
    sess = tf.compat.v1.Session()
    sess.run(init)
    print("====== init check point ======")
    ckpt = tf.compat.v1.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        Log.add_log("message: load ckpt model:'"+str(ckpt.model_checkpoint_path)+"'")
    else:
        Log.add_log("message:can not find  ckpt model")
        # exit(1)
    print("====== init testing ======")
    test_data = np.zeros([height, width, 3])
    img, _, _, _, _ = deal_img(test_data)
    _ = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
    print("====== yolo initialize complete ======")    
    return

def run(img):
    if img is None:
        return None
    img, nw, nh, img_ori, show_img = deal_img(img)
    if img is None:
        Log.add_log("message:'"+str(img)+"' is None")
        return None

    start = time.perf_counter()
    boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
    end = time.perf_counter()
    # print("time:%f s" %(end-start))
    
    # img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)
    
    # cv2.imshow('img', img_ori)
    # cv2.waitKey(1)
    # print(label ,boxes)
    return label ,boxes

    pass

if __name__ == "__main__":
    Log.add_log("message: into val.main()")
    img = cv2.imread("loss.png")
    run(img)