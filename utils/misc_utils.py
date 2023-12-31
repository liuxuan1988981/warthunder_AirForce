# coding: utf-8

import numpy as np
import tensorflow as tf
import random

def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        print("\rcurrent : {}%".format(i*1.0/len(var_list)*100),end="")
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        # if 'Conv' in var1.name.split('/')[-2]:
        if 'Conv' in var1.name:
            # check type of next layer
            # if 'BatchNorm' in var2.name.split('/')[-2]:
            if 'batch_normalization' in var2.name:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    # print(var_weights)
                    ptr += num_params
                    assign_ops.append(tf.compat.v1.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            # elif 'Conv' in var2.name.split('/')[-2]:
            elif 'bias' in var2.name:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                # print(bias_weights)
                ptr += bias_params
                assign_ops.append(tf.compat.v1.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            # print(var_weights)
            ptr += num_params
            assign_ops.append(
                tf.compat.v1.assign(var1, var_weights, validate_shape=True))
            i += 1
        else:
            print("no")
    print("ptr:%d, weights: %d" %(ptr, len(weights)))

    return assign_ops
