#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
pid=$$
BUILD=build/examples/FRCNN/loc_test_frcnn.bin

$BUILD --gpu $gpu \
    --model matlab/FRCNN/For_LOC/two/res101/test.proto \
    --weights matlab/FRCNN/For_LOC/two/res101/res101_faster_rcnn_final.caffemodel \
    --default_c matlab/FRCNN/For_LOC/two/trecvid.json \
    --image_root matlab/FRCNN/For_LOC/LOC/  \
    --image_list matlab/FRCNN/For_LOC/dataset/test.list.val \
    --out_file matlab/FRCNN/For_LOC/two/res101/out/2_test_list_res101_${pid}.frcnn \
    --max_per_image 100

CAL_AP=matlab/FRCNN/For_LOC/calculate_trecvid_ap.py

#python $CAL_AP --gt matlab/FRCNN/For_LOC/dataset/trecvid_7.val \
#    --answer matlab/FRCNN/For_LOC/res101/out/trecvid_7_test_res101_${pid}.frcnn \
#    --overlap 0.5
