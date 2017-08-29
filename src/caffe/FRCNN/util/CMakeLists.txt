set(FRCNN_util_sources
  frcnn_bbox.cpp
  frcnn_bbox_transform.cpp
  frcnn_config.cpp
  frcnn_file.cpp
  frcnn_nms_kernel.cu
  frcnn_vis.cpp
  frcnn_param.cpp)
ADD_LIBRARY(FRCNN_util ${FRCNN_util_sources})
TARGET_LINK_LIBRARIES(FRCNN_util ${Caffe_LINK})
