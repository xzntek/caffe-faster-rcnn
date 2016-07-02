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
    global_pool = L.Pooling(layer, name = 'global_pool', pool=P.Pooling.AVE, global_pooling=True)

    fc = L.InnerProduct(global_pool, name = 'fc', ex_top = ['fc'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],num_output=10,
            bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
    #acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])

    return to_proto(loss, acc)

##def Data
def prepare_data(lmdb, mean_file, batch_size=100, train=True):
    if train==True:
        data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                name = 'data', ex_top = ['data','label'],
                transform_param=dict(mean_file=mean_file, crop_size=28, mirror=True), include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    else:
        data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                name = 'data', ex_top = ['data','label'],
                transform_param=dict(mean_file=mean_file, crop_size=28), include=dict(phase=getattr(caffe_pb2, 'TEST')))

    #return data, label
    return data, label

def write_prototxt(proto_name, name, protos):
    print('Proto Name : {}'.format(name))
    with open(proto_name, 'w') as model:
        model.write('name: %s\n' % (name))
        for proto in protos:
            model.write('{}'.format(proto))

class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, net_prototxt_path='train_val.prototxt', snapshot = '"snapshot"'):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.1'
        self.sp['momentum'] = '0.9'

        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '200'

        # looks:
        self.sp['display'] = '100'
        self.sp['snapshot'] = '2000'
        self.sp['snapshot_prefix'] = '"' + snapshot + '"' # string withing a string!

        # learning rate policy
        self.sp['lr_policy'] = '"multistep"'
        self.sp['stepvalue'] = ['35000', '55000', '70000']

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0001'
        self.sp['net'] = '"' + net_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '70000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25'  # this has to do with the display.
        self.sp['iter_size'] = '1'  # this is for accumulating gradients

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                if isinstance(value, list):
                    num = len(value)
                    for ii in xrange(num):
                        vv = value[ii]
                        #print 'Type : {}, value: {}'.format(type(vv), vv)
                        if not(type(vv) is str):
                            raise TypeError('All solver parameters must be strings')
                        else:
                            f.write('%s: %s\n' % (key, vv))
                else:
                    raise TypeError('All solver parameters must be strings')
            else:
                f.write('%s: %s\n' % (key, value))


