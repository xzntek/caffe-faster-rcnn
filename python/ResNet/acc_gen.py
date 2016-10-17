#!/usr/bin/env python
import _init_paths
from special_net_spec import layers as L, params as P, to_proto
import caffe
import sys, os, argparse
import os.path as osp
import wrap #this contains some tools that we need
import wrap_identity_acc
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
    parser.add_argument('--crop_size', dest='crop_size',
                        help='crop_size',
                        default=28, type=int)

    parser.add_argument('--ACC_model_1', dest='acc1',
                        help='accalerate model for first convolution',
                        default='bn', type=str)

    parser.add_argument('--ACC_model_2', dest='acc2',
                        help='accalerate model for second convolution',
                        default='bn', type=str)

    parser.add_argument('--Single_Act', dest='single',
                        help='activate function for single 1x1 convolution',
                        default='bn', type=str)

    parser.add_argument("--FirstTrue", action="store_true", dest="first")
    parser.add_argument("--FirstFalse", action="store_false", dest="first")
    parser.set_defaults(first=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args   = parse_args()

    layer_num  = args.resnet_N*6+2
    print 'Resnet N: %d, Layer: %d' %(args.resnet_N, layer_num)

    cifar10_dir = '{}{}_F{}_{}_{}_A{}'.format(args.model, layer_num, args.first, args.acc1, args.acc2, args.single)
    print 'cifar10_dir : {}'.format(cifar10_dir)
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
    snapshot_prefix = osp.join(snapshot, 'cifar10_res{}'.format(layer_num)) ;
    solverprototxt = wrap.CaffeSolver(net_prototxt_path = trainval_proto, snapshot = snapshot_prefix);
    solverprototxt.write(solver_file)

    train_data, train_label = wrap.prepare_data(args.lmdb_train, args.mean_file, args.batch_size_train, True, args.crop_size)
    test_data, test_label = wrap.prepare_data(args.lmdb_test, args.mean_file, args.batch_size_test, False, args.crop_size)

    assert args.acc1 == 'bn' or args.acc1 == 'pn' or args.acc1 == 'none'
    assert args.acc2 == 'bn' or args.acc2 == 'pn' or args.acc2 == 'none'

    #caffemodel = wrap_for_acc.resnet_cifar_acc(test_data, test_label, args.resnet_N)
    main_proto = wrap_identity_acc.resnet_cifar_acc(test_data, test_label, args.resnet_N, PACC = [args.acc1, args.acc2], PAddtioni = args.single, First_Acc = args.first)
    name = '"CIFAR10_Resnet_%d"' % (args.resnet_N*6+2)
    wrap.write_prototxt(trainval_proto, name, [to_proto(train_data, train_label), main_proto])

    shell = osp.join(cifar10_dir, 'train_{}.sh'.format(layer_num))
    with open(shell, 'w') as shell_file:
        shell_file.write('{}\nGLOG_log_dir={} build/tools/caffe train --solver {} --gpu $gpu'.format(wrap.gpu_shell_string(), log, solver_file))

    shell = osp.join(cifar10_dir, 'dis_{}.sh'.format(layer_num))
    weights = osp.join('{}_iter_{}.caffemodel'.format(snapshot_prefix, solverprototxt.sp['max_iter']))
    with open(shell, 'w') as shell_file:
        shell_file.write('{}\nmodel={}\n'.format(wrap.gpu_shell_string(), trainval_proto))
        shell_file.write('weights={}\niters=100\n'.format(weights))
        shell_file.write('./build/tools/display_resnset_sparse --gpu $gpu --model $model --weights $weights --iterations $iters 2>&1 | tee {}'.format(osp.join(log, 'display_$$.log')))

    shell = osp.join(cifar10_dir, 'time_{}.sh'.format(layer_num))
    with open(shell, 'w') as shell_file:
        shell_file.write('{}\nmodel={}\n'.format(wrap.gpu_shell_string(), trainval_proto))
        shell_file.write('weights={}\niters=50\n'.format(weights))
        shell_file.write('OMP_NUM_THREADS=1 ./build/tools/time_for_forward --gpu $gpu --model $model --weights $weights --iterations $iters 2>&1 | tee {}'.format(osp.join(log, 'time_$$.log')))

    ignore = osp.join(cifar10_dir, '.gitignore')
    with open(ignore, 'w') as ignore_file:
        ignore_file.write('*')

    print 'Generate Done, Save in %s' % trainval_proto
