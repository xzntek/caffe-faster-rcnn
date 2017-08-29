subdirs(util)
set(FRCNN_sources
  frcnn_anchor_target_layer.cpp
  frcnn_anchor_target_layer.cu
  frcnn_proposal_layer.cpp
  frcnn_proposal_layer.cu
  frcnn_proposal_target_layer.cpp
  frcnn_proposal_target_layer.cu
  frcnn_roi_data_layer.cpp
  frcnn_roi_data_layer.cu
  frcnn_vid_data_layer.cpp
  frcnn_vid_data_layer.cu
  roi_mask_pooling_layer.cpp
  roi_mask_pooling_layer.cu
  roi_pooling_layer.cpp
  roi_pooling_layer.cu
  smooth_L1_loss_layer.cpp
  smooth_L1_loss_layer.cu)
ADD_LIBRARY(FRCNN ${FRCNN_sources})
TARGET_LINK_LIBRARIES(FRCNN FRCNN_util ${Caffe_LINK})
