#!/usr/bin/env python
import _init_paths
from special_net_spec import layers as L, params as P, to_proto
import caffe
import sys, os, argparse
import os.path as osp
import wrap #this contains some tools that we need

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate cifar 10 resnet prototxt')
    parser.add_argument('--lmdb_train', dest='lmdb_train',
                        help='train lmdb data',
                        default='examples/cifar10/cifar10_train_lmdb', type=str)
    parser.add_argument('--lmdb_test', dest='lmdb_test',
                        help='test lmdb data',
                        default='examples/cifar10/cifar10_test_lmdb', type=str)
    parser.add_argument('--mean_file', dest='mean_file',
                        help='overlap value',
                        default='examples/cifar10/mean.binaryproto', type=str)
    parser.add_argument('--N', dest='resnet_N',
                        help='resnet_N',
                        default=9, type=int)
    parser.add_argument('--batch_size_train', dest='batch_size_train',
                        help='batch_size_train',
                        default=256, type=int)
    parser.add_argument('--batch_size_test', dest='batch_size_test',
                        help='batch_size_test',
                        default=100, type=int)
    parser.add_argument('--model', dest='model',
                        help='model proto dir',
                        default='examples/cifar10_resnet', type=str)
    parser.add_argument('--restype', dest='type',
                        help='resnet type',
                        default='original', type=str)
    # support only two type original and identity
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args   = parse_args()

    cifar10_dir = args.model
    print 'cifar10_dir : {}'.format(args.model)
    layer_num  = args.resnet_N*6+2
    print 'Resnet N: %d, Layer: %d' %(args.resnet_N, layer_num)
    #Make Dir
    if not os.path.isdir(cifar10_dir):
        os.makedirs(cifar10_dir)

    snapshot = osp.join(cifar10_dir, 'snapshot', 'cifar10_res{}'.format(layer_num))
    if not os.path.isdir(snapshot):
        os.makedirs(snapshot)

    log = osp.join(cifar10_dir,'log')
    if not os.path.isdir(log):
        os.makedirs(log)

    trainval_proto = osp.join(cifar10_dir, 'cifar10_res{}_trainval.proto'.format(layer_num))

    #Generate solver
    solver_file = osp.join(cifar10_dir, 'solver_{}.proto'.format(layer_num))
    solverprototxt = wrap.CaffeSolver(net_prototxt_path = trainval_proto)
    solverprototxt.sp['display'] = '100'
    solverprototxt.sp['base_lr'] = '0.1'
    solverprototxt.sp['weight_decay'] = '0.0001'
    solverprototxt.sp['lr_policy'] = '"multistep"'
    solverprototxt.sp['stepvalue'] = ['32000', '48000', '64000']
    solverprototxt.sp['max_iter'] = '64000'
    solverprototxt.sp['test_interval'] = '200'
    solverprototxt.sp['snapshot'] = '4000'
    solverprototxt.sp['snapshot_prefix'] = '"' + snapshot + '"'
    solverprototxt.write(solver_file)

    train_data, train_label = wrap.prepare_data(args.lmdb_train, args.mean_file, args.batch_size_train, True)
    test_data, test_label = wrap.prepare_data(args.lmdb_test, args.mean_file, args.batch_size_test, False)


    if args.type=='original':
        print 'original resnet : %s' % args.type
        caffemodel = wrap.resnet_cifar_ori(test_data, test_label, args.resnet_N)
    elif args.type=='identity':
        print 'identity-mapping resnet : %s\n' % args.type
        caffemodel = wrap.resnet_cifar_pro(test_data, test_label, args.resnet_N)
    else:
        TypeError('Resnet type must be original or identity')
    
    name  = '"CIFAR10_Resnet_%d"' % (args.resnet_N*6+2)
    print 'Name: %s' % name
    with open(trainval_proto, 'w') as model:
        model.write('name: %s\n' % (name))
        model.write('%s\n' % to_proto(train_data, train_label))
        model.write('%s\n' % caffemodel)

    shell = osp.join(cifar10_dir, 'train_{}.sh'.format(layer_num))
    with open(shell, 'w') as shell_file:
        shell_file.write('GLOG_log_dir={} build/tools/caffe train --solver {} --gpu $1'.format(log, solver_file))

    print 'Generate Done'

