#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

CAFFE=build/tools/caffe 

time GLOG_log_dir=matlab/FRCNN/For_LOC/eight/res152_rpn/log $CAFFE train   \
    --gpu $gpu \
    --solver matlab/FRCNN/For_LOC/eight/res152_rpn/solver.proto \
    --weights matlab/FRCNN/For_LOC/eight/res152_rpn/ResNet-152-model.caffemodel
