#!/usr/bin/env python
import _init_paths
from special_net_spec import layers as L, params as P, to_proto
import caffe
import sys, os, argparse
import os.path as osp
import wrap #this contains some tools that we need
sys.setrecursionlimit(1000000)

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
    parser.add_argument('--N', dest='resnet_layer',
                        help='resnet_layer',
                        default=9, type=int)
    parser.add_argument('--FC_N', dest='fc_n',
                        help='fc layer dims',
                        default=9, type=int)
    parser.add_argument('--batch_size_train', dest='batch_size_train',
                        help='batch_size_train',
                        default=256, type=int)
    parser.add_argument('--batch_size_test', dest='batch_size_test',
                        help='batch_size_test',
                        default=100, type=int)
    parser.add_argument('--crop_size', dest='crop_size',
                        help='crop_size',
                        default=28, type=int)
    parser.add_argument('--model', dest='model',
                        help='model proto dir',
                        default='examples/cifar10_resnet', type=str)
    # support only two type original and identity
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args   = parse_args()

    layer_num  = args.resnet_layer
    cifar_dir = '{}_{}'.format(args.model, layer_num)
    print 'cifar_dir : {}'.format(cifar_dir)

    #Make Dir
    if not os.path.isdir(cifar_dir):
        os.makedirs(cifar_dir)

    snapshot = osp.join(cifar_dir, 'snapshot')
    if not os.path.isdir(snapshot):
        os.makedirs(snapshot)

    log = osp.join(cifar_dir,'log')
    if not os.path.isdir(log):
        os.makedirs(log)

    trainval_proto = osp.join(cifar_dir, 'cifar_res{}_trainval.proto'.format(layer_num))

    #Generate warmup solver
    solver_warmup_file = osp.join(cifar_dir, 'solver_{}.warmup'.format(layer_num))
    snapshot_warm_prefix = osp.join(snapshot, 'cifar_res{}_warmup'.format(layer_num))
    solver_warm_prototxt = wrap.CaffeSolver(net_prototxt_path = trainval_proto, snapshot = snapshot_warm_prefix)
    solver_warm_prototxt.sp['snapshot'] = '600'
    solver_warm_prototxt.sp['base_lr'] = '0.01'
    solver_warm_prototxt.sp['max_iter'] = '600'
    solver_warm_prototxt.write(solver_warmup_file)
    #Generate solver
    solver_file = osp.join(cifar_dir, 'solver_{}.proto'.format(layer_num))
    snapshot_prefix = osp.join(snapshot, 'cifar_res{}'.format(layer_num))
    solverprototxt = wrap.CaffeSolver(net_prototxt_path = trainval_proto, snapshot = snapshot_prefix)
    solverprototxt.write(solver_file)

    train_data, train_label = wrap.prepare_data(args.lmdb_train, args.mean_file, args.batch_size_train, True, args.crop_size)
    test_data, test_label = wrap.prepare_data(args.lmdb_test, args.mean_file, args.batch_size_test, False, args.crop_size)

    caffemodel = wrap.resnet_identity_mapping(test_data, test_label, args.resnet_layer, args.fc_n)
    
    name  = '"CIFAR_Resnet_%d"' % (args.resnet_layer)
    data_proto = to_proto(train_data, train_label)
    wrap.write_prototxt(trainval_proto, name, [data_proto, caffemodel])

    # train shell
    shell = osp.join(cifar_dir, 'train_{}.sh'.format(layer_num))
    with open(shell, 'w') as shell_file:
        shell_file.write('{}\n./build/tools/caffe train --solver {} --gpu $gpu 2>&1 | tee {}/warmup.log'.format(wrap.gpu_shell_string(), solver_warmup_file, log))
        W = osp.join('{}_iter_{}.caffemodel'.format(snapshot_warm_prefix, solver_warm_prototxt.sp['max_iter']))
        shell_file.write('\n\nGLOG_log_dir={} ./build/tools/caffe train --solver {} --gpu $gpu --weights {}'.format(log, solver_file, W))

    # count forward time shell
    shell = osp.join(cifar_dir, 'time_{}.sh'.format(layer_num))
    weights = osp.join('{}_iter_{}.caffemodel'.format(snapshot_prefix, solverprototxt.sp['max_iter']))
    with open(shell, 'w') as shell_file:
        shell_file.write('{}\nmodel={}\n'.format(wrap.gpu_shell_string(), trainval_proto))
        shell_file.write('weights={}\niters=50\n'.format(weights))
        shell_file.write('OMP_NUM_THREADS=1 GLOG_log_dir={} ./build/tools/time_for_forward --gpu $gpu --model $model --weights $weights --iterations $iters'.format(log))

    # git ignore file
    ignore = osp.join(cifar_dir, '.gitignore')
    with open(ignore, 'w') as ignore_file:
        ignore_file.write('*')

    print 'Generate Done'

