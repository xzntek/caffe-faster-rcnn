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

    snapshot = osp.join(cifar10_dir, 'snapshot')
    if not os.path.isdir(snapshot):
        os.makedirs(snapshot)

    log = osp.join(cifar10_dir,'log')
    if not os.path.isdir(log):
        os.makedirs(log)

    trainval_proto = osp.join(cifar10_dir, 'cifar10_res{}_trainval.proto'.format(layer_num))

    #Generate solver
    solver_file = osp.join(cifar10_dir, 'solver_{}.proto'.format(layer_num))
    snapshot_prefix = osp.join(snapshot, 'cifar10_res{}'.format(layer_num))
    solverprototxt = wrap.CaffeSolver(net_prototxt_path = trainval_proto, snapshot = snapshot_prefix)
    solverprototxt.write(solver_file)

    train_data, train_label = wrap.prepare_data(args.lmdb_train, args.mean_file, args.batch_size_train, True)
    test_data, test_label = wrap.prepare_data(args.lmdb_test, args.mean_file, args.batch_size_test, False)


    if args.type=='original':
        print 'original resnet : %s' % args.type
        caffemodel = wrap.resnet_cifar_ori(test_data, test_label, args.resnet_N)
    elif args.type=='identity':
        print 'identity-mapping resnet : %s' % args.type
        caffemodel = wrap.resnet_cifar_pro(test_data, test_label, args.resnet_N)
    else:
        TypeError('Resnet type must be original or identity')
    
    name  = '"CIFAR10_Resnet_%d"' % (args.resnet_N*6+2)
    data_proto = to_proto(train_data, train_label)
    wrap.write_prototxt(trainval_proto, name, [data_proto, caffemodel])

    shell = osp.join(cifar10_dir, 'train_{}.sh'.format(layer_num))
    with open(shell, 'w') as shell_file:
        shell_file.write('GLOG_log_dir={} build/tools/caffe train --solver {} --gpu $1'.format(log, solver_file))

    shell = osp.join(cifar10_dir, 'time_{}.sh'.format(layer_num))
    weights = osp.join('{}_iter_{}.caffemodel'.format(snapshot_prefix, solverprototxt.sp['max_iter']))
    with open(shell, 'w') as shell_file:
        shell_file.write('gpu=$1\nmodel={}\n'.format(trainval_proto))
        shell_file.write('weights={}\niters=50\n'.format(weights))
        shell_file.write('OMP_NUM_THREADS=1 ./build/tools/time_for_forward --gpu $gpu --model $model --weights $weights --iterations $iters 2>&1 | tee {}'.format(osp.join(cifar10_dir,'time_$$.log')))

    ignore = osp.join(cifar10_dir, '.gitignore')
    with open(ignore, 'w') as ignore_file:
        ignore_file.write('*')

    print 'Generate Done'

