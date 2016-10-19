from __future__ import print_function
from wrap import Conv2D, BN, BN_ReLU, shortcut_block, caffe_pb2
from wrap import prepare_data, gpu_shell_string, write_prototxt
from special_net_spec import layers as L, params as P, to_proto

## Shorcut Method
shortcutType = 'C'  # Use Conv
shortcutType = 'B'  # ImageNet
shortcutType = 'A'  # Cifar10

def Acc_BN_ReLU(name, input):
    temp = BN(name, input)
    return L.ReLU(temp, name = name+'-relu', ex_top = [name+'-relu'], in_place = False)

def acc_until(input, name, num_output, stride, kernel_size, pad):
    conv = Conv2D(input = input, name = name, stride = stride, kernal_size = kernel_size, pad = pad, num_output = num_output)
    acc  = Conv2D(input = input, name = name+'-acc', stride = stride, kernal_size = kernel_size, pad = pad, num_output = 1)
    acc  = Acc_BN_ReLU(input = acc, name = name+'-acc')
    acc  = L.Tile(acc, name = name+'-acc-tile', ex_top=[name+'-acc-tile'],tile_param=dict(tiles=num_output))
    return L.Eltwise(conv, acc, name = name+'-acc-prod', ex_top = [name+'-acc-prod'], operation=P.Eltwise.PROD)

def bottleneck_acc(input, name, in_channel, out_channel, stride = 1, first = False):
    nBottleneckPlane = out_channel / 4;
    if in_channel == out_channel: # -- most Residual Units have this shape
        identity = input
        # conv1x1
        conv = BN_ReLU(input = input, name = name+'-1')
        conv = Conv2D(input = conv, name = name+'-1', stride = stride, kernal_size = 1, pad = 0, num_output = nBottleneckPlane)
        # conv3x3
        conv = BN_ReLU(input = conv, name = name+'-2')
        conv = Conv2D(input = conv, name = name+'-2', stride = 1, kernal_size = 3, pad = 1, num_output = nBottleneckPlane)
        # conv1x1
        conv = BN_ReLU(input = conv, name = name+'-3')
        conv = Conv2D(input = conv, name = name+'-3', stride = 1, kernal_size = 1, pad = 0, num_output = out_channel)

        # short cut
        return L.Eltwise(conv, identity, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)
    else: # -- Residual Units for increasing dimensions
        block = BN_ReLU(input = input, name = name+'-0')
        # conv1x1
        conv = Conv2D(input = block, name = name+'-1', stride = stride, kernal_size = 1, pad = 0, num_output = nBottleneckPlane)
        # conv3x3
        conv = BN_ReLU(input = conv, name = name+'-2')
        conv = Conv2D(input = conv, name = name+'-2', stride = 1, kernal_size = 3, pad = 1, num_output = nBottleneckPlane)
        # conv1x1
        conv = BN_ReLU(input = conv, name = name+'-3')
        conv = Conv2D(input = conv, name = name+'-3', stride = 1, kernal_size = 1, pad = 0, num_output = out_channel)
        # shortcut
        shortcut = Conv2D(input = block, name = name+'-shortcut', stride = stride, kernal_size = 1, pad = 0, num_output = out_channel)
        # shortcut = shortcut_block(block, name = name+'-shortcut', in_channel = in_channel, out_channel = out_channel, stride = stride); ## 'B' Type
        return L.Eltwise(conv, shortcut, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)

def basic_acc(input, name, in_channel, out_channel, stride = 1, first = False):
    block = BN_ReLU(input = input, name = name+'-1')
    if first:
        input = block
    ##block = Conv2D(input = block, name = name+'-1', stride = stride, kernal_size = 3, pad = 1, num_output = out_channel)
    block = acc_until(input = block, name = name+'-1', num_output = out_channel, stride = stride, kernel_size = 3, pad = 1)
    block = BN_ReLU(input = block, name = name+'-2')
    #block = Conv2D(input = block, name = name+'-2', stride = 1, kernal_size = 3, pad = 1, num_output = out_channel)
    block = acc_until(input = block, name = name+'-2', num_output = out_channel, stride = 1, kernel_size = 3, pad = 1)
    ## CIFAR 10
    shortcut = shortcut_block(input = input, name = name+'-shortcut', in_channel = in_channel, out_channel = out_channel, stride = stride)
    return L.Eltwise(block, shortcut, name = name+'-sum', ex_top = [name+'-sum'], operation=P.Eltwise.SUM)
    
# Generate resnet cifar10 train && test prototxt. n_size control number of layers.
# The total number of layers is  6 * n_size + 2. Here I don't know any of implementation 
# which can contain simultaneously TRAIN && TEST phase. 
# ==========================Note here==============================
# !!! SO YOU have to include TRAIN && TEST by your own AFTER you use the script to generate the prototxt !!!
def identity_layer(input, name, in_channel, out_channel, count, stride, layer_block, first = False):
    layer = layer_block(input, name+'.0', in_channel, out_channel, stride = stride)
    for i in xrange(1, count):
        layer = layer_block(layer, name+'.{}'.format(i), out_channel, out_channel, stride = 1)
    return layer

def resnet_identity_mapping(data, label, depth, fc_n, bottleneck):
    assert((depth - 2) % 9 == 0), 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)'
    print('Pre-Activation ResNet-{} CIFAR-10'.format(depth))
    if bottleneck:
        print('Prefer bottleneck structure : 16 64 128 256')
        assert ((depth-2) % 9 == 0)
        n = (depth - 2) / 9
        nStages = [16, 64, 128, 256]
        layer_block = bottleneck_acc
    else:
        print('Prefer bottleneck structure : 16 16 32 64')
        assert ((depth-2) % 6 == 0)
        n = (depth - 2) / 6
        nStages = [16, 16, 32, 64]
        layer_block = basic_acc
    layer = Conv2D(input = data, name = 'init', num_output = nStages[0]) #-- one conv at the beginning (spatial size: 32x32)
    layer = identity_layer(input = layer, name = 'res1', in_channel = nStages[0], out_channel = nStages[1], count = n, stride = 1, layer_block = layer_block, first = True)#-- Stage 1 (spatial size: 32x32)
    layer = identity_layer(input = layer, name = 'res2', in_channel = nStages[1], out_channel = nStages[2], count = n, stride = 2, layer_block = layer_block) #-- Stage 2 (spatial size: 16x16)
    layer = identity_layer(input = layer, name = 'res3', in_channel = nStages[2], out_channel = nStages[3], count = n, stride = 2, layer_block = layer_block) #-- Stage 3 (spatial size: 8x8)
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
