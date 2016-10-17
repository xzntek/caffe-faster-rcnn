from __future__ import print_function
import numpy as np
from special_net_spec import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

# helper function for building ResNet block structures 
# The function below does computations: bottom--->conv--->BatchNorm
def Conv2D(name, input, num_output, kernal_size = 3, pad = 1, weight_filler = dict(type='xavier'), stride = 1, bias_filler = dict(type='constant',value=0), bias_term = False):
    param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)]
    if bias_term == True:
        return L.Convolution(input, name=name, ex_top = [name], kernel_size=kernal_size, pad = pad, stride = stride, num_output=num_output, weight_filler=weight_filler, bias_filler=bias_filler)
    else:
        return L.Convolution(input, name=name, ex_top = [name], kernel_size=kernal_size, pad = pad, stride = stride, num_output=num_output, weight_filler=weight_filler, bias_term=False)

def conv_factory(bottom, ks, n_out, name, stride=1, pad=0):
    conv = L.Convolution(bottom, name = name+'-conv', ex_top = [name+'-conv'], kernel_size=ks, stride=stride, num_output=n_out, pad=pad,
            param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    batch_norm = L.BatchNorm(conv, name = name+'-bn', in_place=True,
            param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, name = name+'-scale', bias_term=True, in_place=True)
    return scale

# bottom--->conv--->BatchNorm--->ReLU
def conv_factory_relu(bottom, ks, n_out, name, stride=1, pad=0):
    conv = L.Convolution(bottom, name = name+'-conv', ex_top = [name+'-conv'], kernel_size=ks, stride=stride, num_output=n_out, pad=pad,
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    batch_norm = L.BatchNorm(conv, name = name+'-bn', in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, name = name+'-scale', bias_term=True, in_place=True)
    relu = L.ReLU(scale, name = name+'-relu', in_place=True)
    return relu

def BatchNorm(name, input, Has_Scale = True):
    temp = L.BatchNorm(input, name = name+'-bn', ex_top = [name+'-bn'], param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    if Has_Scale:
        #return L.Scale(temp, name = name+'-scale', in_place = True, scale_param = dict(bias_term=True), bias_filler = dict(type='constant',value=0))
        return L.Scale(temp, name = name+'-scale', in_place = True, bias_term=True)
    else:
        return temp

def BN_ReLU(name, input, Has_Scale = True):
    temp = BatchNorm(name, input, Has_Scale = Has_Scale )
    return L.ReLU(temp, name = name+'-relu', in_place = True)

#  Residual building block! Implements option (A) from Section 3.3. The input
 #  is passed through two 3x3 convolution layers. Currently this block only supports 
 #  stride == 1 or stride == 2. When stride is 2, the block actually does pooling.
 #  Instead of simply doing pooling which may cause representational bottlneck as
 #  described in inception v3, here we use 2 parallel branches P && C and add them
 #  together. Note pooling branch may has less channels than convolution branch so we
 #  need to do zero-padding along channel dimension. And to the best knowledge of
 #  ours, we haven't found current caffe implementation that supports this operation. 
 #  So later I'll give implementation in C++ and CUDA.
def original_residual_block(bottom, name, num_filters, stride=1):
    if stride == 1:
        conv1 = conv_factory_relu(bottom, 3, num_filters, name+'-1', 1, 1)
        conv2 = conv_factory(conv1, 3, num_filters, name+'-2', 1, 1)
        add = L.Eltwise(bottom, conv2, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)
        return L.ReLU(add, name = name+'_relu', in_place=True)
    elif stride == 2:
        conv1 = conv_factory_relu(bottom, 3, num_filters, name+'-1', 2, 1)
        conv2 = conv_factory(conv1, 3, num_filters, name+'-2', 1, 1)
        pool = L.Pooling(bottom, name = name+'-pool', ex_top = [name+'-pool'], pool=P.Pooling.AVE, kernel_size=2, stride=2)
        pad = L.PadChannel(pool, name = name+'-pad', ex_top = [name+'-pad'], num_channels_to_pad=num_filters / 2)
        add = L.Eltwise(conv2, pad, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)
        return L.ReLU(add, name = name+'_relu', in_place=True)
    else:
        raise Exception('Currently, stride must be either 1 or 2.')

def proposed_residual_block(input, name, out_channel, stride = 1, First = False):
    if First == False:
        layer = BN_ReLU(input = input, name = name+'-1')
    else:
        layer = input
    conv1 = Conv2D(input = layer, name = name+'-1-conv', stride = stride, num_output = out_channel)
    bn2   = BN_ReLU(input = conv1, name = name+'-2')
    conv2 = Conv2D(input = bn2, name = name+'-2-conv', num_output = out_channel)
    if stride != 1:
        #input = self.Conv2D(input = input, name = name+'-A1-conv', kernal_size = 3, stride = stride, pad = 1, num_output = out_channel)
        input = L.Pooling(input, name = name+'-pool', ex_top = [name+'-pool'], pool=P.Pooling.AVE, kernel_size=2, stride=2)
        input = L.PadChannel(input, name = name+'-pad', ex_top = [name+'-pad'], num_channels_to_pad=out_channel / 2)

    return L.Eltwise(conv2, input, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)

def bottleneck_block(input, name, in_channel, out_channel, stride = 1):
    nBottleneckPlane = out_channel / 4;
    if in_channel == out_channel: # -- most Residual Units have this shape
        identity = input
        # conv1x1
        conv = BN_ReLU(input = input, name = name+'-1')
        conv = Conv2D(input = conv, name = name+'-1-conv', stride = stride, kernal_size = 1, pad = 0, num_output = nBottleneckPlane)
        # conv3x3
        conv = BN_ReLU(input = conv, name = name+'-2')
        conv = Conv2D(input = conv, name = name+'-2-conv', stride = 1, kernal_size = 3, pad = 1, num_output = nBottleneckPlane)
        # conv1x1
        conv = BN_ReLU(input = conv, name = name+'-3')
        conv = Conv2D(input = conv, name = name+'-3-conv', stride = 1, kernal_size = 1, pad = 0, num_output = out_channel)

        # short cut
        return L.Eltwise(conv, identity, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)
    else: # -- Residual Units for increasing dimensions
        block = BN_ReLU(input = input, name = name+'-0')
        # conv1x1
        conv = Conv2D(input = block, name = name+'-1-conv', stride = stride, kernal_size = 1, pad = 0, num_output = nBottleneckPlane)
        # conv3x3
        conv = BN_ReLU(input = conv, name = name+'-2')
        conv = Conv2D(input = conv, name = name+'-2-conv', stride = 1, kernal_size = 3, pad = 1, num_output = nBottleneckPlane)
        # conv1x1
        conv = BN_ReLU(input = conv, name = name+'-3')
        conv = Conv2D(input = conv, name = name+'-3-conv', stride = 1, kernal_size = 1, pad = 0, num_output = out_channel)
        # shortcut
        shortcut = Conv2D(input = block, name = name+'-0-conv', stride = stride, kernal_size = 1, pad = 0, num_output = out_channel)
        return L.Eltwise(conv, shortcut, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)


# Generate resnet cifar10 train && test prototxt. n_size control number of layers.
# The total number of layers is  6 * n_size + 2. Here I don't know any of implementation 
# which can contain simultaneously TRAIN && TEST phase. 
# ==========================Note here==============================
# !!! SO YOU have to include TRAIN && TEST by your own AFTER you use the script to generate the prototxt !!!
def resnet_cifar_ori(data, label, n_size=3):

    residual = conv_factory_relu(data, 3, 16, 'init', 1, 1)
    # --------------> 16, 32, 32    1st group
    for i in xrange(n_size):
        residual = original_residual_block(bottom = residual, name = 'res1.{}'.format(i), num_filters = 16)

    # --------------> 32, 16, 16    2nd group
    residual = original_residual_block(bottom = residual, name = 'res2.0', num_filters = 32, stride = 2)
    for i in xrange(n_size - 1):
        residual = original_residual_block(bottom = residual, name = 'res2.{}'.format(i+1), num_filters = 32)

    # --------------> 64, 8, 8      3rd group
    residual = original_residual_block(bottom = residual, name = 'res3.0', num_filters = 64, stride = 2)
    for i in xrange(n_size - 1):
        residual = original_residual_block(bottom = residual, name = 'res3.{}'.format(i+1), num_filters = 64)

    # -------------> end of residual
    global_pool = L.Pooling(residual, name = 'global_pool', ex_top = ['global_pool'], pool=P.Pooling.AVE, global_pooling=True)

    fc = L.InnerProduct(global_pool, name = 'fc', ex_top = ['fc'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],num_output=10,
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
    #acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])
    return to_proto(loss, acc)

def resnet_cifar_pro(data, label, n_size=3):
    # Convolution 0
    layer = Conv2D(input = data, name = 'init', num_output = 16)
    layer = BN_ReLU(input = layer, name = 'init')
    layer = proposed_residual_block(input = layer, name = 'res1.0', stride = 1, out_channel = 16, First = True)

    #32*32, c=16
    for i in xrange(n_size - 1):
        layer = proposed_residual_block(input = layer, name = 'res1.{}'.format(i+1), stride = 1, out_channel = 16)

    #16*16, c=32
    layer = proposed_residual_block(input = layer, name = 'res2.0', stride = 2, out_channel = 32)
    for i in xrange(n_size - 1):
        layer = proposed_residual_block(input = layer, name = 'res2.{}'.format(i+1), stride = 1, out_channel = 32)

    #8*8, c=64
    layer = proposed_residual_block(input = layer, name = 'res3.0', stride = 2, out_channel = 64)
    for i in xrange(n_size - 1):
        layer = proposed_residual_block(input = layer, name = 'res3.{}'.format(i+1), stride = 1, out_channel = 64)

    #After Last Res Unit, with a BN and ReLU
    layer = BN_ReLU(input = layer, name = 'final-post')

    ## Ave Pooling
    global_pool = L.Pooling(layer, name = 'global_pool', ex_top = ['global_pool'], pool=P.Pooling.AVE, global_pooling=True)

    fc = L.InnerProduct(global_pool, name = 'fc', ex_top = ['fc'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],num_output=10,
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
    #acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])

    return to_proto(loss, acc)

def identity_layer(input, name, in_channel, out_channel, count, stride):
    layer = bottleneck_block(input, name+'.0', in_channel, out_channel, stride = stride)
    for i in xrange(1, count):
        layer = bottleneck_block(layer, name+'.{}'.format(i), out_channel, out_channel, stride = 1)
    return layer

def resnet_identity_mapping(data, label, depth, fc_n):
    assert((depth - 2) % 9 == 0), 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)'
    n = (depth - 2) / 9
    print(' | ResNet-{} CIFAR-10'.format(depth))
    nStages = [16, 64, 128, 256]
    layer = Conv2D(input = data, name = 'init', num_output = nStages[0]) #-- one conv at the beginning (spatial size: 32x32)
    layer = identity_layer(input = layer, name = 'res1', in_channel = nStages[0], out_channel = nStages[1], count = n, stride = 1) #-- Stage 1 (spatial size: 32x32)
    layer = identity_layer(input = layer, name = 'res2', in_channel = nStages[1], out_channel = nStages[2], count = n, stride = 2) #-- Stage 2 (spatial size: 16x16)
    layer = identity_layer(input = layer, name = 'res3', in_channel = nStages[2], out_channel = nStages[3], count = n, stride = 2) #-- Stage 3 (spatial size: 8x8)
    #After Last Res Unit, with a BN and ReLU
    layer = BN_ReLU(input = layer, name = 'final-post')
    ## Ave Pooling
    global_pool = L.Pooling(layer, name = 'global_pool', pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(global_pool, name = 'fc', ex_top = ['fc'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)], num_output=fc_n,
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
    acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])
    return to_proto(loss, acc)


##def Data
def prepare_data(lmdb, mean_file, batch_size=100, train=True, crop_size=28):
    if train==True:
        data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                name = 'data', ex_top = ['data','label'],
                transform_param=dict(mean_file=mean_file, crop_size=crop_size, mirror=True), include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    else:
        data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                name = 'data', ex_top = ['data','label'],
                transform_param=dict(mean_file=mean_file, crop_size=crop_size), include=dict(phase=getattr(caffe_pb2, 'TEST')))

    #return data, label
    return data, label

def write_prototxt(proto_name, name, protos):
    print('Proto Name : {}'.format(name))
    with open(proto_name, 'w') as model:
        model.write('name: %s\n' % (name))
        for proto in protos:
            model.write('{}'.format(proto))

def gpu_shell_string():
    A = 'if [ ! -n "$1" ] ;then\n'
    B = '\techo "\\$1 is empty, default is 0"\n'
    C = '\tgpu=0\nelse\n'
    D = '\techo "use $1-th gpu"\n\tgpu=$1\nfi'
    return '{}{}{}{}'.format(A,B,C,D)
