# This shell give an example to show how to set params in cifar10_resnet_model.py
# cifar10_resnet_model.py is a python shell to generate resnet 
# Using the "Identity Mappings"
# n = {3,5,7,9,18,27} for layers = {20,32,44,56,110,164}
#
# Details in the following two papers.
# [Deep Residual Learning for Image Recognition](http://arxiv.org/pdf/1512.03385v1.pdf)
# [Identity Mappings in Deep Residual Networks](http://arxiv.org/pdf/1603.05027v2.pdf)
#
# Please execute this shell in the caffe root dir
#python ./python/ResNet/cifar10_resnet_model.py \
#    --lmdb_train examples/cifar10/cifar10_train_lmdb \
#    --lmdb_test  examples/cifar10/cifar10_test_lmdb  \
#    --mean_file  examples/cifar10/mean.binaryproto   \
#    --N          3 \
#    --batch_size_train 128 \
#    --batch_size_test  100 \
#    --model      examples/cifar10_resnet/cifar10_resnet

# original get 91.07% accuracy, identity get 91.51% accuracy [32*32]--crop-->[28*28]
# batch size 64 for 4-gpus
python ./python/ResNet/cifar10_resnet_model.py --N 3 --restype original --model examples/original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb \
    --lmdb_test examples/cifar10/cifar10_test_lmdb --batch_size_train 64 --batch_size_test 100 --crop_size 32
python ./python/ResNet/cifar10_resnet_model.py --N 3 --restype identity --model examples/identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb \
    --lmdb_test examples/cifar10/cifar10_test_lmdb --batch_size_train 64 --batch_size_test 100 --crop_size 32

# 4 pixels are PADded on each sides
# original [20] 91.89% accuracy, identity [20] 91.79% accuracy [36*36]--crop-->[32*32]
# original [32] 92.17% accuracy, identity [32] 92.65% accuracy
# original [44] 92.61% accuracy, identity [44] 92.90% accuracy 
# original [56] 93.11% accuracy, identity [56] 93.32% accuracy
# original [110] 93.53% accuracy, identity [110] 94.41% accuracy, 
# use bottleneck struct
# identity [164] 94.36% accuracy [warm up]
python ./python/ResNet/cifar10_resnet_model.py --N 3 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 5 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 7 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 9 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 11 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 13 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 15 --restype original --model examples/PAD_4_original_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
# For identity Mapping 
python ./python/ResNet/cifar10_resnet_model.py --N 3 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 5 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 7 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 9 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 11 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 13 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/cifar10_resnet_model.py --N 15 --restype identity --model examples/PAD_4_identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 64 --batch_size_test 100 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto

# For 164 1001, we need bottleneck struct
# Cifar 100 [56 layers]:69.34%,  3*3->3*3
# [110 layers]:72.71%
python ./python/ResNet/identity_mapping.py --N 164 --FC_N 10 --model examples/Identity_resnet_cifar10 --lmdb_train examples/cifar10/cifar10_train_lmdb_pad_4 \
    --lmdb_test examples/cifar10/cifar10_test_lmdb_pad_4 --batch_size_train 16 --batch_size_test 50 --crop_size 32 --mean_file examples/cifar10/pad_4_mean.binaryproto
python ./python/ResNet/identity_mapping.py --N 164 --FC_N 100 --model examples/Identity_resnet_cifar100 --lmdb_train examples/cifar100/cifar100_train_lmdb_pad_4 \
    --lmdb_test examples/cifar100/cifar100_test_lmdb_pad_4 --batch_size_train 16 --batch_size_test 50 --crop_size 32 --mean_file examples/cifar100/pad_4_mean.binaryproto
