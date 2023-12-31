# coding:utf-8
# training own net

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
from src.Data import Data
from src.YOLO import YOLO
from os import path
import config
import time
import numpy as np
from src import Log
from src import Optimizer
from src import Learning_rate as Lr
from src.Loss import Loss

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1


width = config.width
height = config.height
batch_size = config.batch_size
class_num = config.class_num
anchors =  np.asarray(config.anchors).astype(np.float32).reshape([-1, 3, 2])

iou_thresh = config.iou_thresh
prob_thresh = config.prob_thresh
score_thresh = config.score_thresh

weight_decay = config.weight_decay
cls_normalizer = config.cls_normalizer
iou_normalizer = config.iou_normalizer

lr_type = config.lr_type
lr_init = config.lr_init
lr_lower = config.lr_lower
piecewise_boundaries = config.piecewise_boundaries
piecewise_values = config.piecewise_values
optimizer_type = config.optimizer_type
momentum = config.momentum

data_debug = config.data_debug
model_name = config.voc_model_name
model_path = config.voc_model_path
total_epoch = config.total_epoch
save_per_epoch = config.save_per_epoch
# model_path = "./checkpoint/"
# model_name = "model"
name_file = './data/train.names'                # dataset's classfic names
train_file = './data/train.txt'
# val_dir = "./test_pic"  # Test folder directory, which stores test pictures
# save_dir = "./save"         # the folder to save result image

# compute current  epoch : tensor
def compute_curr_epoch(global_step, batch_size, imgs_num):
    '''
    global_step: current step
    batch_size:batch_size
    imgs_num: total images number
    '''
    epoch = global_step * batch_size / imgs_num
    return  tf.cast(epoch, tf.int32)

# training
def backward():
    yolo = YOLO()
    data = Data(train_file, name_file, class_num, batch_size, anchors, width=width, height=height, data_debug=data_debug)
    
    imgs_ls = data.imgs_path
    labels_ls = data.labels_path
    print(imgs_ls[:5])
    print(labels_ls[:5])
    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices((imgs_ls, labels_ls))
    dataset = dataset.shuffle(len(imgs_ls))   # shuffle
    dataset = dataset.batch(batch_size=batch_size)    # 
    dataset = dataset.map(
        lambda imgs_batch, xmls_batch: tf.py_func(
                data.load_tf_batch_data, 
                inp=[(imgs_batch, xmls_batch)],
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=4
    )
    dataset = dataset.prefetch(20)
    # repeat
    dataset = dataset.repeat()
    # iterator
    iterator = dataset.make_initializable_iterator()
    inputs, y1_true, y2_true, y3_true = iterator.get_next()
    # set shape
    inputs.set_shape([None, None, None, 3])
    y1_true.set_shape([batch_size, None, None, 3, 5+class_num])
    y2_true.set_shape([batch_size, None, None, 3, 5+class_num])
    y3_true.set_shape([batch_size, None, None, 3, 5+class_num])

    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, class_num=class_num, weight_decay=weight_decay, isTrain=True)

    global_step = tf.Variable(0, trainable=False)
    
    # loss value of yolov4
    loss = Loss().yolo_loss([feature_y1, feature_y2, feature_y3], 
                                                        [y1_true, y2_true, y3_true], 
                                                        [anchors[2], anchors[1], anchors[0]], 
                                                        width, height, class_num,
                                                        cls_normalizer=cls_normalizer,
                                                        iou_normalizer=iou_normalizer,
                                                        iou_thresh=iou_thresh, 
                                                        prob_thresh=prob_thresh, 
                                                        score_thresh=score_thresh)
    l2_loss = tf.compat.v1.losses.get_regularization_loss()
    
    epoch = compute_curr_epoch(global_step, batch_size, len(data.imgs_path))
    lr = Lr.config_lr(lr_type, lr_init, lr_lower=lr_lower, \
                                        piecewise_boundaries=piecewise_boundaries, \
                                        piecewise_values=piecewise_values, epoch=epoch)
    optimizer = Optimizer.config_optimizer(optimizer_type, lr, momentum)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(loss+l2_loss)
        clip_grad_var = [gv if gv[0] is None else[tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_step = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    # initialize
    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        sess.run(iterator.initializer)
        step = 0
        
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            step = eval(step)
            Log.add_log("message: load ckpt model, global_step=" + str(step))
        else:
            Log.add_log("message: can not find ckpt model")
        
        curr_epoch = step // data.steps_per_epoch
        while curr_epoch < total_epoch:
            for _ in range(data.steps_per_epoch):
                start = time.perf_counter()
                _, loss_, step, lr_ = sess.run([train_step, loss, global_step, lr])
                end = time.perf_counter()
                
                if (loss_ > 1e3) and (step > 1e3):
                    Log.add_log("error:loss exception, loss_value = "+str(loss_))
                    ''' break the process or lower learning rate '''
                    raise ValueError("error:loss exception, loss_value = "+str(loss_)+", please lower your learning rate")
                    # lr = tf.math.maximum(tf.math.divide(lr, 10), config.lr_lower)
                
                if step % 5 == 2:
                    print("step: %6d, epoch:%3d, loss: %.5g\t, wh: %s, lr:%.5g\t, time: %5f s"
                                %(step, curr_epoch, loss_, "("+str(width)+","+str(height)+")", lr_, end-start))
                    Log.add_loss(str(step) + "\t" + str(loss_))

            curr_epoch += 1
            if curr_epoch % save_per_epoch == 0:
                # save ckpt model
                Log.add_log("message: save ckpt model, step=" + str(step) +", lr=" + str(lr_))
                saver.save(sess, path.join(model_path, model_name), global_step=step)                  

        # save ckpt model
        Log.add_log("message:save final ckpt model, step=" + str(step))
        saver.save(sess, path.join(model_path, model_name), global_step=step)
        
    return 0


if __name__ == "__main__":
    Log.add_log("message: goto backward function")
    Log.add_loss("###########")
    backward()