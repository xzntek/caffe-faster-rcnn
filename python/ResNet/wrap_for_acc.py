from __future__ import print_function
import numpy as np
from special_net_spec import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
import wrap

global_single = []

# helper function for building ResNet block structures 
def ConvS_SC(name, input, out_channel, stride, Addition):
    conv = L.Convolution(input, name=name+'-single-pre', ex_top = [name+'-single-pre'], kernel_size=1, pad = 0, stride = stride, num_output=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant',value=0))
    if Addition == 'bn':
        conv = wrap.BatchNorm(name=name+'-single-pre', input = conv, Has_Scale = False)
    elif Addition == 'sbn':
        conv = wrap.BatchNorm(name=name+'-single-pre', input = conv, Has_Scale = True)
    elif Addition == 'none':
        pass
    else:
        TypeError('Addition must be bn or sbn or none')
    # Above is conv with bn
    relu = L.ReLU(conv, name = name+'-single-relu', ex_top = [name+'-single'])

    # For Loss
    global_single.append(L.L2Loss(relu, name = name+'-single-relu-L2', ex_top = [name+'-single-relu-L2'], loss_weight = 1))

    mul  = L.SplitConcat(relu, name = name+'-mul', ex_top = [name+'-mul'], convolution_param=dict(num_output=out_channel))
    return mul

def acc_residual_block(input, name, out_channel, stride = 1, ACC = False, Addition = 'none'):
    layer = input

    layer = wrap.BN_ReLU(input = layer, name = name+'-1')
    acc   = ConvS_SC(input = layer, name = name+'-1', out_channel= out_channel, stride = stride, Addition = Addition)
    conv1 = wrap.Conv2D(input = layer, name = name+'-1-conv', stride = stride, num_output = out_channel)
    if ACC == True:
        conv1 = L.Eltwise(acc, conv1, name = name+'-acc1', ex_top = [name+'-acc1'], operation=P.Eltwise.PROD)

    layer = conv1

    acc   = ConvS_SC(input = layer, name = name+'-2', out_channel= out_channel, stride = 1, Addition = Addition)
    layer = wrap.BN_ReLU(input = conv1, name = name+'-2')
    conv2 = wrap.Conv2D(input = layer, name = name+'-2-conv', num_output = out_channel)
    if ACC == True:
        conv2 = L.Eltwise(acc, conv2, name = name+'-acc2', ex_top = [name+'-acc2'], operation=P.Eltwise.PROD)

    if stride != 1:
        #input = self.Conv2D(input = input, name = name+'-A1-conv', kernal_size = 3, stride = stride, pad = 1, num_output = out_channel)
        input = L.Pooling(input, name = name+'-pool', ex_top = [name+'-pool'], pool=P.Pooling.AVE, kernel_size=2, stride=2)
        input = L.PadChannel(input, name = name+'-pad', ex_top = [name+'-pad'], num_channels_to_pad=out_channel / 2)

    return L.Eltwise(conv2, input, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)

def resnet_cifar_acc(data, label, n_size=3):
    # Convolution 0
    layer = wrap.Conv2D(input = data, name = 'init', num_output = 16)
    layer = L.BatchNorm(layer, name = 'init-bn', in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    input = layer = L.ReLU(layer, name = 'init-relu', in_place=True)

    layer = wrap.Conv2D(input = input, name = 'res0.0-1-conv', num_output = 16)
    layer = wrap.BN_ReLU(input = layer, name = 'res0.0-1-bn')
    layer = wrap.Conv2D(input = layer, name = 'res0.0-2-conv', num_output = 16)
    layer = L.Eltwise(layer, input, name = 'res0.0-sum', ex_top = ['res0.0-sum'], operation=P.Eltwise.SUM)

    # From Last To Top
    ACC = [True, True, True, True, True]
    Addition = ['none', 'none', 'none', 'none', 'none']
    #Addition = ['bn', 'bn', 'bn', 'bn', 'bn']
    #Addition = ['sbn', 'sbn', 'sbn', 'sbn', 'sbn']
    
    #32*32, c=16
    for_Addition = Addition.pop()
    for_Acc = ACC.pop()
    for i in xrange(n_size - 1):
        layer = acc_residual_block(input = layer, name = 'res1.{}'.format(i+1), stride = 1, out_channel = 16, ACC = for_Acc, Addition = for_Addition)

    #16*16, c=32
    layer = acc_residual_block(input = layer, name = 'res2.0', stride = 2, out_channel = 32, ACC = for_Acc, Addition = Addition.pop())
    for_Addition = Addition.pop()
    for_Acc = ACC.pop()
    for i in xrange(n_size - 1):
        layer = acc_residual_block(input = layer, name = 'res2.{}'.format(i+1), stride = 1, out_channel = 32, ACC = for_Acc, Addition = for_Addition)

    #8*8, c=64
    layer = acc_residual_block(input = layer, name = 'res3.0', stride = 2, out_channel = 64, ACC=ACC.pop(), Addition = Addition.pop())
    for_Addition = Addition.pop()
    for_Acc = ACC.pop()
    for i in xrange(n_size - 1):
        layer = acc_residual_block(input = layer, name = 'res3.{}'.format(i+1), stride = 1, out_channel = 64, ACC = for_Acc, Addition = for_Addition)

    #After Last Res Unit, with a BN and ReLU
    layer = wrap.BN_ReLU(input = layer, name = 'final-post')

    ## Ave Pooling
    global_pool = L.Pooling(layer, name = 'global_pool', pool=P.Pooling.AVE, global_pooling=True)

    fc = L.InnerProduct(global_pool, name = 'fc', ex_top = ['fc'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],num_output=10,
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
    #acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])
    
    return acc, loss
    #return to_proto(loss, acc)
