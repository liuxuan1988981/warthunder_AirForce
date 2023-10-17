# coding:utf-8
# module of net

import tensorflow as tf
from src.Activation import activation as act
from src.Activation import activation_fn
# slim = tf.contrib.slim

# conv2d
def conv1(inputs, out_channels, kernel_size=3, stride=1):
    '''
    inputs: tensor
    out_channels: output channels  int
    kernel_size: kernel size int
    stride: int
    return:tensor
    ...
    conv2d:
        input : [batch, height, width, channel]
        kernel : [height, width, in_channels, out_channels]
    '''
    # fixed edge of tensor
    if stride > 1:
        inputs = padding_fixed(inputs, kernel_size)
    # 
    inputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride, 
                                                padding=('SAME' if stride == 1 else 'VALID'))  
    return inputs

def get_weight(shape,regularizer):
	#随机生成w
	w = tf.Variable(tf.random.truncated_normal(shape,stddev = 0.1), name="Conv")
	# todo:加入正则化
	# if regularizer != None:
	# 	tf.compat.v1.add_to_collection('losses',tf.compat.v1.contrib.layers.l2_regularizer(regularizer)(w))
	return w

#偏置
def get_bias(shape, stddev=0.1):
    # b = tf.Variable(tf.zeros(shape))
    b = tf.Variable(tf.random.truncated_normal(shape,stddev=stddev), name="bias")   # , seed=2222
    return b

# BN
def batch_norm(x, momentum, train, epsilon=1e-5):
    return tf.compat.v1.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=train)

# 卷积
def conv(inputs, out_channels, kernel_size=3, stride=1, use_bn=True, 
                            decay=0.00036, isTrain=True, activation=act.LEAKY_RELU, 
                            momentum=0.9, alpha=0.1, use_bias=True):
    # in_channels = inputs._shape_as_list()[-1]
    in_channels = inputs.get_shape().as_list()[-1]
    # print(type(kernel_size), type(in_channels), type(out_channels), out_channels)
    w = get_weight([kernel_size, kernel_size, in_channels, out_channels], decay)
    strides = [1, stride, stride, 1]
    # fixed padding
    if stride > 1:
        inputs = padding_fixed(inputs, kernel_size)
    x = tf.nn.conv2d(inputs, w, strides=strides, padding=('SAME' if stride == 1 else 'VALID'))
    if use_bn:
        x = batch_norm(x, momentum=momentum, train=isTrain)
    else:
        if use_bias:
            bias = get_bias([out_channels])
            x = tf.nn.bias_add(x, bias)
    if activation:
        x = activation_fn(x, activation, alpha=alpha)
    return x

# fixed edge of tensor
def padding_fixed(inputs, kernel_size):
    '''
    padding zeros around edge
    '''
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    inputs = tf.pad(inputs, [[0,0], [pad_start, pad_end], [pad_start, pad_end], [0,0]])
    return inputs

# implement residual block of yolov4
def yolo_res_block(inputs, in_channels, res_num, double_ch=False, activation=act.MISH):
    '''
    implement residual block of yolov4
    inputs: tensor
    res_num: run res_num  residual block
    '''
    out_channels = in_channels
    if double_ch:
        out_channels = in_channels * 2

    # 3,1,r,1 block
    net = conv(inputs, in_channels*2, stride=2, activation=activation)            
    route = conv(net, out_channels, kernel_size=1, activation=activation)     # in_channels
    net = conv(net, out_channels, kernel_size=1, activation=activation)# in_channels
    
    # 1,3,s block
    for _ in range(res_num):
        tmp = net
        net = conv(net, in_channels, kernel_size=1, activation=activation)
        net = conv(net, out_channels, activation=activation)                                  # in_channels
        # add:shortcut layer
        net = tmp + net

    # 1,r,1 block
    net = conv(net, out_channels, kernel_size=1, activation=activation)       # in_channels
    # concat:route layer
    net = tf.concat([net, route], -1)
    net = conv(net, in_channels*2, kernel_size=1, activation=activation)
    
    return net

# conv block that kernel is 3*3 and 1*1 
def yolo_conv_block(net,in_channels, a, b, activation=act.LEAKY_RELU):
    '''
    net: tensor
    a: the number of conv is a and the kernel size is interleaved 1*1 and 3*3 
    b: number of 1*1 convolution
    '''
    for _ in range(a):
        out_channels = int(in_channels / 2)
        net = conv(net, out_channels, kernel_size=1, activation=activation)
        net = conv(net, in_channels, activation=activation)
    
    out_channels = in_channels
    for _ in range(b):
        out_channels = int(out_channels / 2)
        net = conv(net, out_channels, kernel_size=1, activation=activation)

    return net

# spp maxpool block
def yolo_maxpool_block(inputs):
    '''
    spp
    inputs:[N, 19, 19, 512]
    return:[N, 19, 19, 2048]
    '''
    max_5 = tf.nn.max_pool(inputs, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    max_9 = tf.nn.max_pool(inputs, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    max_13 = tf.nn.max_pool(inputs, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    # concat
    net = tf.concat([max_13, max_9, max_5, inputs], -1)
    return net

# output width and height are twice of input
def yolo_upsample_block(inputs, in_channels, route, activation):
    '''
    inputs:  tensor
    route: tensor
    '''
    shape = tf.shape(inputs)
    out_height, out_width = shape[1]*2, shape[2]*2
    inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (out_height, out_width))
    
    route = conv(route, in_channels, kernel_size=1, activation=activation)

    net = tf.concat([route, inputs], -1)
    return net

def extraction_feature(inputs, batch_norm_params, weight_decay):
    '''
    inputs:[N, 416, 416, 3]
    '''
    # ########## downsample module ##########
    # with slim.arg_scope([slim.conv2d], 
    #                         normalizer_fn=slim.batch_norm,
    #                         normalizer_params=batch_norm_params,
    #                         biases_initializer=None,
    #                         activation_fn=lambda x: Activation.activation_fn(x, act.MISH),
    #                         # Important:here you can set the activation function as leaky_relu to save GPU memory
    #                         # activation_fn=lambda x: Activation.activation_fn(x, act.LEAKY_RELU),
    #                         weights_regularizer=slim.l2_regularizer(weight_decay)):
    if True:
        # with tf.variable_scope('Downsample'):
        if True:
            net = conv(inputs, 32)
            # downsample
            # res1
            net = yolo_res_block(net, 32, 1, double_ch=True)    # *2
            # res2
            net = yolo_res_block(net, 64, 2)
            # res8
            net = yolo_res_block(net, 128, 8)
            # features of 54 layer
            # [N, 76, 76, 256]
            up_route_54 = net
            # res8
            net = yolo_res_block(net, 256, 8)
            # features of 85 layer
            # [N, 38, 38, 512]
            up_route_85 = net
            # res4
            net = yolo_res_block(net, 512, 4)

    # ########## leaky_relu ##########
    # with slim.arg_scope([slim.conv2d], 
    #                         normalizer_fn=slim.batch_norm,
    #                         normalizer_params=batch_norm_params,
    #                         biases_initializer=None,
    #                         activation_fn=lambda x: Activation.activation_fn(x, act.LEAKY_RELU, 0.1),
    #                         weights_regularizer=slim.l2_regularizer(weight_decay)):
    if True:
        # with tf.variable_scope('leaky_relu'):
        if True:
            net = yolo_conv_block(net, 1024, 1, 1)
            # pooling:SPP
            # [N, 19, 19, 512] => [N, 19, 19, 2048]
            net = yolo_maxpool_block(net)
            net = conv(net, 512, kernel_size=1, activation=act.LEAKY_RELU)
            net = conv(net, 1024, activation=act.LEAKY_RELU)
            net = conv(net, 512, kernel_size=1, activation=act.LEAKY_RELU)
            # features of 116 layer
            # [N, 19, 19, 512]
            route_3 = net

            net = conv(net, 256, kernel_size=1, activation=act.LEAKY_RELU)
            net = yolo_upsample_block(net, 256, up_route_85, activation=act.LEAKY_RELU)

            net = yolo_conv_block(net, 512, 2, 1)
            # features of 126 layer
            # [N, 38, 38, 256]
            route_2 = net

            # [N, 38, 38, 256] => [N, 38, 38, 128]
            net = conv(net, 128, kernel_size=1, activation=act.LEAKY_RELU)
            net = yolo_upsample_block(net, 128, up_route_54, activation=act.LEAKY_RELU)

            net = yolo_conv_block(net, 256, 2, 1)
            # features of 136 layer
            # [N, 76, 76, 128]
            route_1 = net

    return route_1, route_2, route_3


