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

def ConvS_SC(name, input, out_channel, stride):
    conv = L.Convolution(input, name=name+'-single-pre', ex_top = [name+'-single-pre'], kernel_size=1, pad = 0, stride = stride, num_output=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant',value=0))
    conv = BatchNorm(name=name+'-single-pre', input = conv, Has_Scale = True)
    relu = L.ReLU(conv, name = name+'-single-relu', ex_top = [name+'-single'])
    mul  = L.SplitConcat(relu, name = name+'-mul', ex_top = [name+'-mul'], convolution_param=dict(num_output=out_channel))
    return mul

def acc_residual_block(input, name, out_channel, stride = 1, ACC = False):
    layer = input

    layer = BN_ReLU(input = layer, name = name+'-1')
    acc   = ConvS_SC(input = layer, name = name+'-1', out_channel= out_channel, stride = stride)
    conv1 = Conv2D(input = layer, name = name+'-1-conv', stride = stride, num_output = out_channel)
    if ACC == True:
        conv1 = L.Eltwise(acc, conv1, name = name+'-acc1', ex_top = [name+'-acc1'], operation=P.Eltwise.PROD)

    layer = conv1

    layer = BN_ReLU(input = conv1, name = name+'-2')
    acc   = ConvS_SC(input = layer, name = name+'-2', out_channel= out_channel, stride = 1)
    conv2 = Conv2D(input = layer, name = name+'-2-conv', num_output = out_channel)
    if ACC == True:
        conv2 = L.Eltwise(acc, conv2, name = name+'-acc2', ex_top = [name+'-acc2'], operation=P.Eltwise.PROD)

    if stride != 1:
        #input = self.Conv2D(input = input, name = name+'-A1-conv', kernal_size = 3, stride = stride, pad = 1, num_output = out_channel)
        input = L.Pooling(input, name = name+'-pool', ex_top = [name+'-pool'], pool=P.Pooling.AVE, kernel_size=2, stride=2)
        input = L.PadChannel(input, name = name+'-pad', ex_top = [name+'-pad'], num_channels_to_pad=out_channel / 2)

    return L.Eltwise(conv2, input, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)

def resnet_cifar_acc(data, label, n_size=3):
    # Convolution 0
    layer = Conv2D(input = data, name = 'init', num_output = 16)
    layer = L.BatchNorm(layer, name = 'init-bn', in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    input = layer = L.ReLU(layer, name = 'init-relu', in_place=True)

    layer = Conv2D(input = input, name = 'res0.0-1-conv', num_output = 16)
    layer = BN_ReLU(input = layer, name = 'res0.0-1-bn')
    layer = Conv2D(input = layer, name = 'res0.0-2-conv', num_output = 16)
    layer = L.Eltwise(layer, input, name = 'res0.0-sum', ex_top = ['res0.0-sum'], operation=P.Eltwise.SUM)

    #32*32, c=16
    for i in xrange(n_size - 1):
        layer = acc_residual_block(input = layer, name = 'res1.{}'.format(i+1), stride = 1, out_channel = 16, ACC=True)

    #16*16, c=32
    layer = acc_residual_block(input = layer, name = 'res2.0', stride = 2, out_channel = 32, ACC=True)
    for i in xrange(n_size - 1):
        layer = acc_residual_block(input = layer, name = 'res2.{}'.format(i+1), stride = 1, out_channel = 32, ACC=True)

    #8*8, c=64
    layer = acc_residual_block(input = layer, name = 'res3.0', stride = 2, out_channel = 64, ACC=True)
    for i in xrange(n_size - 1):
        layer = acc_residual_block(input = layer, name = 'res3.{}'.format(i+1), stride = 1, out_channel = 64, ACC=True)

    #After Last Res Unit, with a BN and ReLU
    layer = BN_ReLU(input = layer, name = 'final-post')

    ## Ave Pooling
    global_pool = L.Pooling(layer, name = 'global_pool', pool=P.Pooling.AVE, global_pooling=True)

    fc = L.InnerProduct(global_pool, name = 'fc', ex_top = ['fc'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],num_output=10,
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
    #acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])

    return to_proto(loss, acc)
