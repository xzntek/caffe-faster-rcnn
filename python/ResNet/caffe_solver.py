from __future__ import print_function
import numpy as np
from special_net_spec import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

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
        self.sp['stepvalue'] = ['32000', '48000', '64000']

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0001'
        self.sp['net'] = '"' + net_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '64000'
        self.sp['test_initialization'] = 'true'
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


