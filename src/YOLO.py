# coding:utf-8
# Yolov5的实现

from src.module import  conv
from src.Activation import activation as act
import tensorflow as tf
from src import Log
import math
from src.Yolov5_backbone import Yolov5_backbone
import numpy as np

class YOLO():
    def __init__(self):
        self.backbone = Yolov5_backbone()
        self.init_net_args()    # 初始化网络参数
        pass

    # 初始化网络参数
    def init_net_args(self):
        self.args_head = [
            [1, "self.CBL", [512, 1, 1]],
            [1, "self.Upsample", [2]],
            [1, "self.Concat", [6]],  # cat backbone P4
            [3, "self.C3", [512, False]],  # 13

            [1, "self.CBL", [256, 1, 1]],
            [1, "self.Upsample", [2]],
            [1, "self.Concat", [4]],  # cat backbone P3
            [3, "self.C3", [256, False]],  # 17 (P3/8-small)

            [1, "self.CBL", [256, 3, 2]],
            [1, "self.Concat", [14]],  # cat head P4
            [3, "self.C3", [512, False]],  # 20 (P4/16-medium)

            [1, "self.CBL", [512, 3, 2]],
            [1, "self.Concat", [10]],  # cat head P5
            [3, "self.C3", [1024, False]],  # 23 (P5/32-large)
            #          small         medium     large
            [1, "Detect", [17, 20, 23]],  # Detect(P3, P4, P5)
        ]
        self.channel_scale, self.module_count_scale = [0.50, 0.33]
        pass

    # 构建网络
    def forward(self, inputs, class_num, weight_decay=0.00036, momentum=0.843, isTrain=True):
        self.isTrain = isTrain
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # backbone
        x8, x16, inputs = self.backbone.forward(inputs, 
                                                weight_decay=weight_decay, 
                                                momentum=self.momentum, 
                                                isTrain=isTrain)

        def Detect(x_small, x_medium, x_large):
            # yolo
            x_small = conv(x_small, 3*(5+class_num), 
                                                    kernel_size=1, stride=1, use_bn=False, 
                                                    use_bias=True, activation=None, 
                                                    decay=self.weight_decay,
                                                    momentum=self.momentum)
            x_medium = conv(x_medium, 3*(5+class_num), 
                                                    kernel_size=1, stride=1, use_bn=False, 
                                                    use_bias=True, activation=None, 
                                                    decay=self.weight_decay,
                                                    momentum=self.momentum)
            x_large = conv(x_large, 3*(5+class_num), 
                                                    kernel_size=1, stride=1, use_bn=False, 
                                                    use_bias=True, activation=None, 
                                                    decay=self.weight_decay,
                                                    momentum=self.momentum)
            return x_large, x_medium, x_small

        ls_shortcut = []
        ls_concat = [x16, x8, None, None]
        concat_index = 0
        for index,(num, func, args) in enumerate(self.args_head):
            num = max(round(num*self.module_count_scale), 1) if num > 1 else num
            out_channel = self.make_divisible(args[0] * self.channel_scale)
            args_new = [inputs, out_channel, *args[1:]]
            for _ in range(num):
                # print(func+"("+str(args_new)+")")
                # c3的重复是c3内部的
                if "C3" in func:
                    args_new.insert(2, num)
                elif func.split('.')[-1] in ["Concat"]:
                    args_new.insert(1, ls_concat[concat_index])
                    concat_index += 1
                elif "Detect" in func:
                    args_new = [ls_shortcut[i-10] for i in args]

                inputs = eval(func)(*args_new)
                
                if func.split('.')[-1] in ["SPPF", "Upsample", "C3", "Concat", "Detect"]:
                    # 只执行一次
                    break
            if index == 4:
                ls_concat[2] = inputs
            if index == 0:
                ls_concat[3] = inputs
            
            # 保存shortcut
            ls_shortcut.append(inputs) 
        return inputs

    def CBL(self, inputs, filters, kernel_size=3, 
                    stride=1, use_bn=True, use_bias=True,
                    activation=act.LEAKY_RELU):
        # ! 注意原版用的silu
        ''' 用了bn就不会用bias了 '''
        inputs = conv(inputs, filters, kernel_size=kernel_size, stride=stride, 
                                use_bn=use_bn, isTrain=self.isTrain, 
                                activation=activation, use_bias=use_bias, 
                                decay=self.weight_decay,
                                momentum=self.momentum)
        return inputs
    
    def Bottleneck(self, inputs, filters, shortcut=True, e=0.5):
        c_ = int(filters*e)
        # print("bottleneck : %d", c_)
        in_channels = inputs._shape_as_list()[-1]
        x = self.CBL(inputs, c_, kernel_size=1)
        x = self.CBL(x, filters, kernel_size=3)
        add = ((shortcut) and (in_channels==filters))
        return tf.add(inputs, x) if add else x
    
    def C3(self, inputs, filters, num=1, shortcut=True, e=0.5):
        # 隐藏层channel
        c_ = int(filters*e)
        # print("c3 : {}".format(c_))
        net = self.CBL(inputs, c_, kernel_size=1)

        for _ in range(num):
            net = self.Bottleneck(net, c_, shortcut=shortcut, e=1.0)

        res = self.CBL(inputs, c_, kernel_size=1)
        net = tf.concat([net, res], -1)

        net = self.CBL(net, filters, kernel_size=1)
        return net

    def SPPF(self, inputs, out_channel, k=5):
        c_ = inputs._shape_as_list()[-1] // 2
        inputs = self.CBL(inputs, c_, 1, 1)
        
        max_1 = tf.nn.max_pool(inputs, [1, k, k, 1], [1,1,1,1], padding='SAME')
        max_2 = tf.nn.max_pool(max_1, [1, k, k, 1], [1,1,1,1], padding='SAME')
        max_3 = tf.nn.max_pool(max_2, [1, k, k, 1], [1,1,1,1], padding='SAME')

        net = tf.concat([inputs, max_1, max_2, max_3 ], -1)

        net = self.CBL(net, out_channel, 1, 1)
        return net

    def SPP(self, inputs, filters):
        in_channels = inputs._shape_as_list()[-1]
        c_ = in_channels // 2
        net = self.CBL(inputs, c_, kernel_size=1)
        max_5 = tf.nn.max_pool(net, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
        max_9 = tf.nn.max_pool(net, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
        max_13 = tf.nn.max_pool(net, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
        # concat
        net = tf.concat([net, max_5, max_9, max_13], -1)
        net = self.CBL(net, filters, kernel_size=1)
        return net

    def Upsample(self, inputs, *args, **kw):
        shape = tf.shape(inputs)
        out_height, out_width = shape[1]*2, shape[2]*2
        inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (out_height, out_width))
        return inputs
    
    def Concat(self, x1, x2, *args, **kw):
        return tf.concat([x1, x2], -1)

    # 计算kernel个数,保证为 divisor 的整数倍
    def make_divisible(self, x, divisor=8):
        # Returns x evenly divisible by divisor
        return int(np.math.ceil(x / divisor) * divisor)
